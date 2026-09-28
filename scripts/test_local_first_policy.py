"""Test Local-First Cascade Policy on Canonical Validation Set."""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
    reciprocal_rank_fusion,
)
from avito_candidate_generation.research import (
    context_folds,
    context_frame,
    field_text,
    ground_truth_map,
)
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def haversine_km(lat1, lon1, lat2, lon2):
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = np.sin(dlat/2.0)**2 + np.cos(r_lat1)*np.cos(r_lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c

def main():
    print("Loading data...")
    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0]
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts = val_with_bm_pos[["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category", "internal_query_id"]].drop_duplicates().reset_index(drop=True)
    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    loc_geo = train.dropna(subset=["item_latitude", "item_longitude"]).groupby("item_location_id")[["item_latitude", "item_longitude"]].mean()
    loc_lat_dict = loc_geo["item_latitude"].astype(float).to_dict()
    loc_lon_dict = loc_geo["item_longitude"].astype(float).to_dict()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"]))
    item_lat_map = dict(zip(item_id_list, benchmark_items["item_latitude"].astype(float).fillna(0.0)))
    item_lon_map = dict(zip(item_id_list, benchmark_items["item_longitude"].astype(float).fillna(0.0)))

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))

    # Microcategory Routing Top-5
    print("Fitting routing...")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)

    del train, work, folds, merged, train_part, val_rows, val_with_bm_pos
    gc.collect()

    microcat_to_items = {}
    for item_id in item_id_list:
        mc = item_microcat_map[item_id]
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    allowed_items_per_query = []
    for cats in top5_microcats:
        allowed = set()
        for c in cats:
            allowed.update(microcat_to_items.get(str(c), set()))
        allowed_items_per_query.append(allowed)

    # Lexical: Title word + Title char
    print("Fitting lexical indices...")
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branches=["branch_a", "branch_b"], min_df=2)
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    title_char_index = SparseBranchIndex.fit(item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    title_word_queries = field_text(val_contexts, ("search_query",))
    query_params_texts = field_text(val_contexts, ("search_query", "search_infm_params_text"))

    print("Retrieving lexical candidates...")
    routed_word = field_index.branches["branch_a"].retrieve_routed(title_word_queries, allowed_items_per_query, k=1000)
    routed_char = title_char_index.retrieve_routed(title_word_queries, allowed_items_per_query, k=1000)
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(query_params_texts, allowed_items_per_query, k=1000)
    global_word = field_index.branches["branch_a"].retrieve(title_word_queries, k=300)
    global_char = title_char_index.retrieve(title_word_queries, k=300)

    del field_index, title_char_index
    gc.collect()

    # Dense E5
    print("Running Dense E5...")
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)

    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=1000)
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=300)

    sources = [routed_e5, routed_char, routed_word, routed_title_params, global_e5, global_word, global_char]
    weights = [1.5, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3]

    print("Computing candidate RRF (depth=1000)...")
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    # Local-first cascade policy:
    # Tier 1: same location & same category (ranked by RRF score)
    # Tier 2: nearby distance <= 50km & same category (ranked by RRF score * geo_decay)
    # Tier 3: any location with same category (ranked by RRF score * geo_decay)
    # Tier 4: other categories (fallback)
    for max_local_items in [35, 40, 45, 50]:
        final_rankings = []
        for qid, ranking in zip(query_ids, raw_rrf, strict=True):
            q_loc = query_loc_map[qid]
            q_cat = query_cat_map[qid]
            q_lat = loc_lat_dict.get(q_loc)
            q_lon = loc_lon_dict.get(q_loc)

            tier1 = []  # same location, same category
            tier2 = []  # nearby (<= 50km), same category
            tier3 = []  # same category, other locations
            tier4 = []  # other

            for item_id, score in ranking:
                it_loc = item_loc_map.get(item_id)
                it_cat = item_cat_map.get(item_id)
                it_lat = item_lat_map.get(item_id, 0.0)
                it_lon = item_lon_map.get(item_id, 0.0)

                if it_cat == q_cat:
                    if it_loc == q_loc:
                        tier1.append((item_id, score))
                    elif q_lat is not None and it_lat != 0.0:
                        dist = haversine_km(q_lat, q_lon, it_lat, it_lon)
                        if dist <= 50.0:
                            tier2.append((item_id, score * np.exp(-0.02 * dist)))
                        else:
                            tier3.append((item_id, score * np.exp(-0.05 * dist)))
                    else:
                        tier3.append((item_id, score * 0.1))
                else:
                    tier4.append((item_id, score * 0.001))

            tier1.sort(key=lambda p: (-p[1], p[0]))
            tier2.sort(key=lambda p: (-p[1], p[0]))
            tier3.sort(key=lambda p: (-p[1], p[0]))
            tier4.sort(key=lambda p: (-p[1], p[0]))

            # Combine up to 50 items
            selected = tier1[:max_local_items]
            seen_items = {it for it, _ in selected}
            for candidate_tier in [tier2, tier3, tier4]:
                for it, sc in candidate_tier:
                    if it not in seen_items:
                        selected.append((it, sc))
                        seen_items.add(it)
                    if len(selected) >= 50:
                        break
                if len(selected) >= 50:
                    break

            final_rankings.append(selected[:50])

        m = evaluate_rankings(final_rankings, query_ids, relevant, ks=(50,))
        print(f"Max local items = {max_local_items}: Recall@50 = {m['recall@50']:.4f}")

if __name__ == "__main__":
    main()
