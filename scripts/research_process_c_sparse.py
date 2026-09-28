"""Process C: Sparse Retrieval Experiments (H7: Morphology/Stemming, H8: Params/Numeric, H6: Fallback).

Evaluates:
- Numeric token prevalence in queries and filters
- Word TF-IDF vs Char TF-IDF vs Stemmed Word TF-IDF (SnowballStemmer)
- Params & Numeric matching branch
- Incremental candidate coverage over base pool (word + char + E5)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import snowballstemmer

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
    incremental_union_coverage,
    oracle_candidate_coverage,
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
    extract_canonical_numbers,
    normalize_lexical_text,
    numeric_matching_rank,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def stem_text(text: str, stemmer: Any) -> str:
    words = re.findall(r"[\w]+", text.lower())
    if not words:
        return ""
    stemmed = stemmer.stemWords(words)
    return " ".join(stemmed)


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("PROCESS C: SPARSE EXPERIMENTS (H7: Stemming, H8: Params/Numeric, H6: Fallback)")
    print("=" * 70)

    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0].copy()
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts = (
        val_with_bm_pos[
            [
                "search_query",
                "search_location_id",
                "search_is_delivery_search",
                "search_infm_params_text",
                "search_category",
                "internal_query_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    item_params = [str(x) for x in benchmark_items["item_infm_params_text"].fillna("").tolist()]
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"].astype(int)))

    # 1. Numeric Prevalence
    print("\n--- 1. Numeric Prevalence Audit ---")
    val_queries_has_num = [
        bool(extract_canonical_numbers(q)) for q in val_contexts["search_query"].fillna("")
    ]
    bm_queries_has_num = [
        bool(extract_canonical_numbers(q)) for q in benchmark_queries["search_query"].fillna("")
    ]
    val_params_has_num = [
        bool(extract_canonical_numbers(p)) for p in val_contexts["search_infm_params_text"].fillna("")
    ]
    bm_params_has_num = [
        bool(extract_canonical_numbers(p)) for p in benchmark_queries["search_infm_params_text"].fillna("")
    ]

    print(f"Val queries containing numeric: {np.mean(val_queries_has_num):.4f}")
    print(f"BM queries containing numeric:  {np.mean(bm_queries_has_num):.4f}")
    print(f"Val params containing numeric:  {np.mean(val_params_has_num):.4f}")
    print(f"BM params containing numeric:   {np.mean(bm_params_has_num):.4f}")

    # 2. Routing Setup (K=5)
    print("\n--- 2. Setting up Microcategory Routing (K=5) ---")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)

    microcat_to_items: dict[str, set[str]] = {}
    for item_id, mc in item_microcat_map.items():
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    allowed_items_per_query = []
    for cats in top5_microcats:
        allowed = set()
        for c in cats:
            allowed.update(microcat_to_items.get(str(c), set()))
        allowed_items_per_query.append(allowed)

    # 3. Fitting base branches: Word, Char, Stemmed Word
    print("\n--- 3. Fitting Sparse Indices ---")
    word_index = SparseBranchIndex.fit(
        item_id_list, item_titles, name="title_word", analyzer="word", ngram_range=(1, 2), min_df=2
    )
    char_index = SparseBranchIndex.fit(
        item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2
    )

    stemmer = snowballstemmer.stemmer("russian")
    print("Stemming item titles...")
    stemmed_item_titles = [stem_text(t, stemmer) for t in item_titles]
    stemmed_index = SparseBranchIndex.fit(
        item_id_list, stemmed_item_titles, name="title_stemmed", analyzer="word", ngram_range=(1, 2), min_df=2
    )

    title_queries = field_text(val_contexts, ("search_query",))
    stemmed_queries = [stem_text(q, stemmer) for q in title_queries]

    # 4. Retrieving candidates
    print("\n--- 4. Retrieving candidates ---")
    routed_word = word_index.retrieve_routed(title_queries, allowed_items_per_query, k=1000)
    routed_char = char_index.retrieve_routed(title_queries, allowed_items_per_query, k=1000)
    routed_stemmed = stemmed_index.retrieve_routed(stemmed_queries, allowed_items_per_query, k=1000)

    # Dense E5 candidates for base pool
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)
    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=1000)

    # 5. Measure Standalone Recalls
    print("\n--- 5. Standalone Recall@k ---")
    m_word = evaluate_rankings(routed_word, query_ids, relevant, ks=(50, 100, 200, 500, 1000))
    m_char = evaluate_rankings(routed_char, query_ids, relevant, ks=(50, 100, 200, 500, 1000))
    m_stem = evaluate_rankings(routed_stemmed, query_ids, relevant, ks=(50, 100, 200, 500, 1000))
    m_e5 = evaluate_rankings(routed_e5, query_ids, relevant, ks=(50, 100, 200, 500, 1000))

    print(f"Routed Word:    R@50={m_word['recall@50']:.4f}, R@500={m_word['recall@500']:.4f}, R@1000={m_word['recall@1000']:.4f}")
    print(f"Routed Char:    R@50={m_char['recall@50']:.4f}, R@500={m_char['recall@500']:.4f}, R@1000={m_char['recall@1000']:.4f}")
    print(f"Routed Stemmed: R@50={m_stem['recall@50']:.4f}, R@500={m_stem['recall@500']:.4f}, R@1000={m_stem['recall@1000']:.4f}")
    print(f"Routed Dense E5: R@50={m_e5['recall@50']:.4f}, R@500={m_e5['recall@500']:.4f}, R@1000={m_e5['recall@1000']:.4f}")

    # 6. Incremental Coverage Over Base Pool (Word + Char + E5)
    print("\n--- 6. Incremental Coverage Audit (H7 Stemming Acceptance/Kill) ---")
    base_cov = oracle_candidate_coverage([routed_word, routed_char, routed_e5], query_ids, relevant, ks=(500, 1000))
    cov_base_500 = base_cov["union_coverage@500"]
    cov_base_1000 = base_cov["union_coverage@1000"]

    stem_cov = oracle_candidate_coverage([routed_word, routed_char, routed_e5, routed_stemmed], query_ids, relevant, ks=(500, 1000))
    cov_with_stem_500 = stem_cov["union_coverage@500"]
    cov_with_stem_1000 = stem_cov["union_coverage@1000"]

    delta_stem_500 = cov_with_stem_500 - cov_base_500
    delta_stem_1000 = cov_with_stem_1000 - cov_base_1000

    print(f"Base Pool (Word+Char+E5) Coverage: @500={cov_base_500:.4f}, @1000={cov_base_1000:.4f}")
    print(f"+ Stemmed Coverage:                @500={cov_with_stem_500:.4f} (Δ={delta_stem_500:+.4f}), @1000={cov_with_stem_1000:.4f} (Δ={delta_stem_1000:+.4f})")

    # 7. Incremental Coverage of Numeric Matching (H8)
    print("\n--- 7. Incremental Coverage Audit (H8 Numeric Matching) ---")
    title_and_params = [f"{t} {p}" for t, p in zip(item_titles, item_params, strict=True)]
    numeric_candidates = numeric_matching_rank(
        queries=title_queries,
        item_texts=title_and_params,
        item_ids=item_id_list,
        k=500,
    )
    cov_with_num_1000 = oracle_candidate_coverage([routed_word, routed_char, routed_e5, numeric_candidates], query_ids, relevant, ks=(1000,))["union_coverage@1000"]
    delta_num_1000 = cov_with_num_1000 - cov_base_1000
    print(f"+ Numeric Matching Coverage:       @1000={cov_with_num_1000:.4f} (Δ={delta_num_1000:+.4f})")

    # 8. Incremental Coverage of Global Fallback (H6)
    print("\n--- 8. Incremental Coverage Audit (H6 Global Fallback) ---")
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=300)
    global_word = word_index.retrieve(title_queries, k=300)
    global_char = char_index.retrieve(title_queries, k=300)

    cov_with_fallback_1000 = oracle_candidate_coverage(
        [routed_word, routed_char, routed_e5, global_e5, global_word, global_char],
        query_ids,
        relevant,
        ks=(1000,),
    )["union_coverage@1000"]
    delta_fallback_1000 = cov_with_fallback_1000 - cov_base_1000
    print(f"+ Global Fallback Coverage:        @1000={cov_with_fallback_1000:.4f} (Δ={delta_fallback_1000:+.4f})")

    results = {
        "numeric_prevalence": {
            "val_queries": float(np.mean(val_queries_has_num)),
            "bm_queries": float(np.mean(bm_queries_has_num)),
            "val_params": float(np.mean(val_params_has_num)),
            "bm_params": float(np.mean(bm_params_has_num)),
        },
        "standalone_recall": {
            "routed_word": m_word,
            "routed_char": m_char,
            "routed_stemmed": m_stem,
            "routed_dense_e5": m_e5,
        },
        "incremental_coverage": {
            "base_pool_500": float(cov_base_500),
            "base_pool_1000": float(cov_base_1000),
            "stemmed_delta_500": float(delta_stem_500),
            "stemmed_delta_1000": float(delta_stem_1000),
            "numeric_delta_1000": float(delta_num_1000),
            "global_fallback_delta_1000": float(delta_fallback_1000),
        },
    }

    out_path = Path("artifacts/research_v3/sparse_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    runtime_sec = time.time() - t0
    print(f"\nProcess C completed in {runtime_sec:.1f}s. Artifact saved to {out_path}")


if __name__ == "__main__":
    main()
