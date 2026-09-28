"""Fine-tune fusion weights for maximum Recall@50 on canonical validation set."""

import json
import time
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

    # Routing
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)

    microcat_to_items = {}
    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_list = [str(x) for x in benchmark_items["item_microcat_id"].tolist()]
    for item_id, microcat in zip(item_id_list, item_microcat_list, strict=True):
        microcat_to_items.setdefault(microcat, set()).add(item_id)

    allowed_items_per_query = []
    for cats in top5_microcats:
        allowed = set()
        for c in cats:
            allowed.update(microcat_to_items.get(c, set()))
        allowed_items_per_query.append(allowed)

    # Lexical
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branch_b_analyzer="char_wb", branch_b_ngram_range=(3, 5), min_df=2)
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    title_char_index = SparseBranchIndex.fit(item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    title_word_queries = field_text(val_contexts, ("search_query",))
    query_params_texts = field_text(val_contexts, ("search_query", "search_infm_params_text"))

    routed_word = field_index.branches["branch_a"].retrieve_routed(title_word_queries, allowed_items_per_query, k=500)
    routed_char = title_char_index.retrieve_routed(title_word_queries, allowed_items_per_query, k=500)
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(query_params_texts, allowed_items_per_query, k=500)
    global_word = field_index.branches["branch_a"].retrieve(title_word_queries, k=200)
    global_char = title_char_index.retrieve(title_word_queries, k=200)

    # Dense E5
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)

    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=500)
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=200)

    print("Source preparation complete. Testing RRF grid...")
    best_weights = None
    best_c = None
    best_r50 = 0.0

    sources = [routed_e5, routed_char, routed_word, routed_title_params, global_e5, global_word, global_char]

    grid = [
        (60.0, [1.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.2]),
        (60.0, [1.5, 1.0, 0.8, 0.5, 0.4, 0.2, 0.2]),
        (60.0, [1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.2]),
        (60.0, [1.8, 1.0, 0.6, 0.4, 0.4, 0.2, 0.2]),
        (40.0, [1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.2]),
        (40.0, [1.8, 1.2, 0.8, 0.4, 0.4, 0.2, 0.2]),
        (30.0, [1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.2]),
        (30.0, [1.8, 1.2, 0.8, 0.5, 0.4, 0.2, 0.2]),
        (20.0, [1.5, 1.2, 0.8, 0.5, 0.4, 0.2, 0.2]),
        (20.0, [2.0, 1.2, 0.8, 0.4, 0.4, 0.2, 0.2]),
        (10.0, [1.5, 1.2, 0.8, 0.5, 0.4, 0.2, 0.2]),
        (60.0, [2.0, 1.0, 0.5, 0.3, 0.5, 0.2, 0.2]),
    ]

    for c, weights in grid:
        fused = reciprocal_rank_fusion(sources, weights=weights, c=c, limit=50)
        m = evaluate_rankings(fused, query_ids, relevant, ks=(50,))
        r50 = m["recall@50"]
        print(f"c={c:<4}, weights={weights}: Recall@50 = {r50:.4f}")
        if r50 > best_r50:
            best_r50 = r50
            best_c = c
            best_weights = weights

    print(f"\nOptimal RRF: c={best_c}, weights={best_weights}, Recall@50={best_r50:.4f}")

if __name__ == "__main__":
    main()
