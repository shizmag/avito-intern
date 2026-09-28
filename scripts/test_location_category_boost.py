"""Test impact of Category filter and Location boost on Recall@50 with clean memory management."""

import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from avito_candidate_generation.research import context_frame, context_folds, ground_truth_map, field_text
from avito_candidate_generation.retrievers.routed_lexical import FieldAwareSparseIndex, SparseBranchIndex
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.routing import ExactPosteriorPredictor, GeneralizingClassifierPredictor, HybridMicrocategoryPredictor
from avito_candidate_generation.fusion.candidate_pool import reciprocal_rank_fusion, evaluate_rankings

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
    query_texts = [str(x) for x in val_contexts["search_query"].tolist()]
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"]))

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))

    # Microcategory Routing Top-5
    print("Fitting routing...")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)

    # Free train data from RAM!
    del train, work, folds, merged, train_part, val_rows, val_with_bm_pos
    gc.collect()
    print("Train data freed from memory.")

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

    # Lexical: Title word only first to test
    print("Fitting title word...")
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branches=["branch_a"], min_df=2)
    title_word_queries = field_text(val_contexts, ("search_query",))
    routed_word = field_index.branches["branch_a"].retrieve_routed(title_word_queries, allowed_items_per_query, k=500)
    global_word = field_index.branches["branch_a"].retrieve(title_word_queries, k=200)

    # Free field_index
    del field_index
    gc.collect()

    # Dense E5
    print("Running Dense E5...")
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)

    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=500)
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=200)

    sources = [routed_e5, routed_word, global_e5, global_word]
    weights = [1.5, 1.0, 0.4, 0.3]

    print("Computing base RRF...")
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=500)
    base_m = evaluate_rankings(raw_rrf, query_ids, relevant, ks=(50, 100, 200, 500))
    print(f"Base RRF (No Location/Category prior): {base_m}")

    # Now let's test RRF with Category and Location boosting:
    for loc_mult in [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]:
        boosted_rankings = []
        for qid, ranking in zip(query_ids, raw_rrf, strict=True):
            q_loc = query_loc_map[qid]
            q_cat = query_cat_map[qid]
            boosted = []
            for item_id, score in ranking:
                it_loc = item_loc_map.get(item_id)
                it_cat = item_cat_map.get(item_id)
                mult = 1.0
                if it_cat == q_cat:
                    mult *= 5.0
                else:
                    mult *= 0.01
                if it_loc == q_loc:
                    mult *= loc_mult
                boosted.append((item_id, score * mult))
            boosted.sort(key=lambda p: (-p[1], p[0]))
            boosted_rankings.append(boosted[:50])

        m = evaluate_rankings(boosted_rankings, query_ids, relevant, ks=(50,))
        print(f"Location mult = {loc_mult:<5}: Recall@50 = {m['recall@50']:.4f}")

if __name__ == "__main__":
    main()
