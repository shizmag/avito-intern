"""Comprehensive retrieval and fusion evaluation on canonical validation set (5,347 queries).

Covers:
1. Lexical branches: Global vs Routed (Word, Char, Title+Params, Filters->Params, Numeric)
2. Dense E5: Global vs Routed vs Routed+Fallback
3. Historical intent incremental coverage test (kill criterion)
4. Candidate pool union coverage at @50, @100, @200, @500, @1000
5. Fusion pipeline selection (Weighted RRF) & Recall@50 optimization
6. Monotonicity verification for all retrievers (r50 <= r100 <= r200 <= r500 <= r1000)
7. Slice breakdowns (seen/unseen, filters/no-filters, cat 114/0, query length)
8. Bootstrap 95% CI on Recall@50
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import (
    assert_recall_monotonic,
    evaluate_rankings,
    evaluate_slices,
    incremental_union_coverage,
    oracle_candidate_coverage,
    reciprocal_rank_fusion,
)
from avito_candidate_generation.research import (
    CONTEXT_COLUMNS,
    context_folds,
    context_frame,
    field_text,
    ground_truth_map,
)
from avito_candidate_generation.retrievers.routed_e5 import (
    E5QueryEncoder,
    RoutedDenseE5Retriever,
)
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
    numeric_matching_rank,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def bootstrap_ci(
    rankings: list[list[tuple[str, float]]],
    query_ids: list[str],
    relevant: dict[str, set[str]],
    k: int = 50,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval for Recall@k."""
    per_query_recalls: list[float] = []
    for qid, ranking in zip(query_ids, rankings, strict=True):
        expected = relevant[qid]
        top_ids = {item_id for item_id, _ in ranking[:k]}
        per_query_recalls.append(len(top_ids & expected) / max(len(expected), 1))

    arr = np.array(per_query_recalls, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)
    mean_val = float(np.mean(arr))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def main() -> None:
    started_all = time.time()
    os.makedirs("artifacts/research", exist_ok=True)

    print("=" * 80)
    print("STEP 1: Load data and build canonical validation set")
    print("=" * 80)
    t0 = time.time()
    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = cast(pd.DataFrame, pd.merge(work, folds, on="internal_query_id"))

    # 80% train, 20% validation
    train_part = cast(pd.DataFrame, merged[merged["fold"] != 0].copy())
    val_rows = cast(pd.DataFrame, merged[merged["fold"] == 0].copy())
    val_with_bm_pos = cast(
        pd.DataFrame,
        val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy(),
    )

    context_cols = [*CONTEXT_COLUMNS, "internal_query_id"]
    val_contexts = cast(
        pd.DataFrame,
        val_with_bm_pos[context_cols].drop_duplicates().reset_index(drop=True),
    )
    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    query_texts = [str(x) for x in val_contexts["search_query"].tolist()]
    gt_df = cast(
        pd.DataFrame,
        val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates(),
    )
    relevant = ground_truth_map(gt_df)

    print(
        f"Validation set: {len(val_contexts):,} queries, {sum(len(v) for v in relevant.values()):,} GT pairs. "
        f"Loaded in {time.time() - t0:.1f}s"
    )

    # Load precomputed benchmark item embeddings
    print("\nLoading precomputed E5 benchmark item embeddings...")
    t0 = time.time()
    e5_items_json = [
        str(x)
        for x in json.loads(
            Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text()
        )
    ]
    e5_item_embeddings = np.load(
        "artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r"
    )
    print(
        f"E5 benchmark embeddings loaded: shape {e5_item_embeddings.shape}, dtype {e5_item_embeddings.dtype} in {time.time() - t0:.1f}s"
    )

    print("\n" + "=" * 80)
    print("STEP 2: Microcategory Routing (Top-5 Selected)")
    print("=" * 80)
    t0 = time.time()
    exact = ExactPosteriorPredictor()
    exact.fit(train_part, query_col="search_query", category_col="item_microcat_id")

    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42)
    clf.fit(train_part)

    hybrid = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=clf,
        min_count=1,
        blend_strategy="fallback",
    )
    # Predict Top-5 microcategories for all validation queries
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)
    print(f"Hybrid routing model fit and predicted Top-5 in {time.time() - t0:.1f}s")

    # Map microcategory -> list of item_ids in benchmark_items
    microcat_to_items: dict[str, set[str]] = {}
    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_list = [str(x) for x in benchmark_items["item_microcat_id"].tolist()]
    for item_id, microcat in zip(item_id_list, item_microcat_list, strict=True):
        microcat_to_items.setdefault(microcat, set()).add(item_id)

    allowed_items_per_query: list[set[str]] = []
    for cats in top5_microcats:
        allowed: set[str] = set()
        for c in cats:
            allowed.update(microcat_to_items.get(c, set()))
        allowed_items_per_query.append(allowed)

    containment_values = [
        len(allowed & relevant[qid]) / max(len(relevant[qid]), 1)
        for qid, allowed in zip(query_ids, allowed_items_per_query, strict=True)
    ]
    mean_containment = float(np.mean(containment_values))
    mean_corpus_size = float(np.mean([len(s) for s in allowed_items_per_query]))
    print(
        f"Validation Top-5 Containment: {mean_containment:.4f} (Mean routed corpus: {mean_corpus_size:.0f} items)"
    )

    print("\n" + "=" * 80)
    print("STEP 3: Lexical Retrieval Evaluation (Global vs Routed)")
    print("=" * 80)
    t0 = time.time()
    field_index = FieldAwareSparseIndex.fit(
        benchmark_items,
        branch_b_analyzer="char_wb",
        branch_b_ngram_range=(3, 5),
        min_df=2,
    )
    # Also fit title char_wb separately for clean comparison
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    title_char_index = SparseBranchIndex.fit(
        item_id_list,
        item_titles,
        name="title_char",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    print(f"FieldAwareSparseIndex + TitleChar fitted in {time.time() - t0:.1f}s")

    RETRIEVAL_K = 1000  # Retrieve up to 1000 candidates to verify contract monotonicity
    ks_contract = (50, 100, 200, 500, 1000)

    # 3.1 Global Word (Title)
    print("\n[3.1] Running Global Word (Title)...")
    title_word_queries = field_text(val_contexts, ("search_query",))
    global_word_rankings = field_index.branches["branch_a"].retrieve(
        title_word_queries, k=RETRIEVAL_K
    )
    m_global_word = evaluate_rankings(global_word_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_global_word)
    print(f"Global Word: {m_global_word}")

    # 3.2 Global Char (Title)
    print("\n[3.2] Running Global Char (Title)...")
    global_char_rankings = title_char_index.retrieve(title_word_queries, k=RETRIEVAL_K)
    m_global_char = evaluate_rankings(global_char_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_global_char)
    print(f"Global Char: {m_global_char}")

    # 3.3 Routed Word (Title) - Top-5 Routed
    print("\n[3.3] Running Routed Word (Title, Top-5 routed)...")
    routed_word_rankings = field_index.branches["branch_a"].retrieve_routed(
        title_word_queries, allowed_items_per_query, k=RETRIEVAL_K, global_fallback_k=0
    )
    m_routed_word = evaluate_rankings(routed_word_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_routed_word)
    print(f"Routed Word: {m_routed_word}")

    # 3.4 Routed Char (Title) - Top-5 Routed
    print("\n[3.4] Running Routed Char (Title, Top-5 routed)...")
    routed_char_rankings = title_char_index.retrieve_routed(
        title_word_queries, allowed_items_per_query, k=RETRIEVAL_K, global_fallback_k=0
    )
    m_routed_char = evaluate_rankings(routed_char_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_routed_char)
    print(f"Routed Char: {m_routed_char}")

    # 3.5 Routed Title+Params (Char)
    print("\n[3.5] Running Routed Title+Params (Char)...")
    query_params_texts = field_text(val_contexts, ("search_query", "search_infm_params_text"))
    routed_title_params_rankings = field_index.branches["branch_b"].retrieve_routed(
        query_params_texts, allowed_items_per_query, k=RETRIEVAL_K, global_fallback_k=0
    )
    m_routed_title_params = evaluate_rankings(
        routed_title_params_rankings, query_ids, relevant, ks=ks_contract
    )
    assert_recall_monotonic(m_routed_title_params)
    print(f"Routed Title+Params: {m_routed_title_params}")

    # 3.6 Routed Filters -> Params (Word)
    print("\n[3.6] Running Routed Filters -> Params (Word)...")
    filters_texts = field_text(val_contexts, ("search_infm_params_text",))
    routed_filters_params_rankings = field_index.retrieve_routed(
        filters_texts,
        allowed_items_per_query,
        k=RETRIEVAL_K,
        global_fallback_k=0,
        branch="branch_d",
    )
    m_routed_filters = evaluate_rankings(
        routed_filters_params_rankings, query_ids, relevant, ks=ks_contract
    )
    assert_recall_monotonic(m_routed_filters)
    print(f"Routed Filters->Params: {m_routed_filters}")

    # 3.7 Numeric Matching
    print("\n[3.7] Running Numeric Matching...")
    all_query_num_text = [
        f"{q} {p}".strip()
        for q, p in zip(title_word_queries, filters_texts, strict=True)
    ]
    all_item_num_text = [
        f"{t} {p}".strip()
        for t, p in zip(
            benchmark_items["item_title_raw"].fillna("").astype(str),
            benchmark_items["item_infm_params_text"].fillna("").astype(str),
            strict=True,
        )
    ]
    numeric_rankings = numeric_matching_rank(
        all_query_num_text, all_item_num_text, item_id_list, k=RETRIEVAL_K
    )
    m_numeric = evaluate_rankings(numeric_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_numeric)
    print(f"Numeric Matching: {m_numeric}")

    # 3.8 Best Lexical Fusion
    print("\n[3.8] Running Best Lexical Fusion (RRF)...")
    lexical_candidates_sources = [
        routed_word_rankings,
        routed_char_rankings,
        routed_title_params_rankings,
        routed_filters_params_rankings,
        global_word_rankings,
    ]
    best_lexical_fusion = reciprocal_rank_fusion(
        lexical_candidates_sources,
        weights=[1.0, 1.0, 0.8, 0.4, 0.3],
        c=60.0,
        limit=RETRIEVAL_K,
    )
    m_best_lexical = evaluate_rankings(best_lexical_fusion, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_best_lexical)
    print(f"Best Lexical Fusion: {m_best_lexical}")

    print("\n" + "=" * 80)
    print("STEP 4: Dense E5 Retrieval Evaluation (Global vs Routed vs Routed+Fallback)")
    print("=" * 80)
    query_emb_cache_path = Path("artifacts/research/val_query_e5_embeddings.npy")
    if query_emb_cache_path.is_file():
        print(f"Loading cached validation query embeddings from {query_emb_cache_path}...")
        val_query_embeddings = np.load(query_emb_cache_path)
    else:
        print("Encoding validation queries with multilingual-e5-base...")
        encoder = E5QueryEncoder()
        t0 = time.time()
        val_query_embeddings = encoder.encode(query_texts, batch_size=64)
        print(f"Encoded {len(query_texts)} queries in {time.time() - t0:.1f}s")
        np.save(query_emb_cache_path, val_query_embeddings)

    e5_retriever = RoutedDenseE5Retriever(
        item_ids=e5_items_json,
        item_embeddings=e5_item_embeddings,
    )

    # 4.1 Global E5
    print("\n[4.1] Running Global E5...")
    t0 = time.time()
    global_e5_rankings = e5_retriever.retrieve_global(val_query_embeddings, k=RETRIEVAL_K)
    print(f"Global E5 retrieval done in {time.time() - t0:.1f}s")
    m_global_e5 = evaluate_rankings(global_e5_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_global_e5)
    print(f"Global E5: {m_global_e5}")

    # 4.2 Routed E5
    print("\n[4.2] Running Routed E5 (Top-5 routed)...")
    t0 = time.time()
    routed_e5_rankings = e5_retriever.retrieve_routed(
        val_query_embeddings, allowed_items_per_query, k=RETRIEVAL_K, global_fallback_k=0
    )
    print(f"Routed E5 retrieval done in {time.time() - t0:.1f}s")
    m_routed_e5 = evaluate_rankings(routed_e5_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_routed_e5)
    print(f"Routed E5: {m_routed_e5}")

    # 4.3 Routed E5 + Global Fallback
    print("\n[4.3] Running Routed E5 + Global Fallback (routed 500 + global 100)...")
    t0 = time.time()
    routed_e5_fallback_rankings = e5_retriever.retrieve_routed(
        val_query_embeddings, allowed_items_per_query, k=RETRIEVAL_K, global_fallback_k=100
    )
    print(f"Routed E5 + Fallback done in {time.time() - t0:.1f}s")
    m_routed_e5_fallback = evaluate_rankings(
        routed_e5_fallback_rankings, query_ids, relevant, ks=ks_contract
    )
    assert_recall_monotonic(m_routed_e5_fallback)
    print(f"Routed E5 + Fallback: {m_routed_e5_fallback}")

    print("\n" + "=" * 80)
    print("STEP 5: Historical Intent Branch Evaluation (Kill Criterion)")
    print("=" * 80)
    from avito_candidate_generation.research_runner import _historical_intent_rankings

    t0 = time.time()
    hist_rankings, hist_info = _historical_intent_rankings(
        train,
        val_contexts,
        query_ids,
        relevant,
        benchmark_items,
        k=RETRIEVAL_K,
        batch_size=64,
        seed=42,
        train_fold=0,
        folds=5,
    )
    print(f"Historical intent retrieval done in {time.time() - t0:.1f}s: {hist_info}")
    m_hist = evaluate_rankings(hist_rankings, query_ids, relevant, ks=ks_contract)
    assert_recall_monotonic(m_hist)
    print(f"Historical Intent Standalone: {m_hist}")

    # Incremental coverage test over (Routed Lexical + Routed E5)
    base_pool = [best_lexical_fusion, routed_e5_fallback_rankings]
    delta_cov500 = incremental_union_coverage(
        base_pool, hist_rankings, query_ids, relevant, k=500
    )
    print(f"Historical intent incremental coverage@500 over Lexical+E5: {delta_cov500:+.6f}")
    if delta_cov500 < 0.005:
        print("KILL CRITERION TRIGGERED: Historical intent adds negligible gain (< 0.005). REJECTED.")
        hist_decision = "REJECT"
    else:
        print("Historical intent provides meaningful incremental coverage. ACCEPTED.")
        hist_decision = "USE"

    print("\n" + "=" * 80)
    print("STEP 6: Candidate Pool Coverage (Oracle Union without Truncation)")
    print("=" * 80)
    oracle_lexical = oracle_candidate_coverage([best_lexical_fusion], query_ids, relevant, ks=ks_contract)
    oracle_dense = oracle_candidate_coverage([routed_e5_fallback_rankings], query_ids, relevant, ks=ks_contract)

    combined_sources = [
        best_lexical_fusion,
        routed_e5_fallback_rankings,
        global_word_rankings,
        global_e5_rankings,
    ]
    oracle_combined = oracle_candidate_coverage(combined_sources, query_ids, relevant, ks=ks_contract)

    print(f"Oracle Best Lexical Coverage: {oracle_lexical}")
    print(f"Oracle Best Dense Coverage:   {oracle_dense}")
    print(f"Oracle Combined Coverage:     {oracle_combined}")

    print("\n" + "=" * 80)
    print("STEP 7: Final Ranking & Fusion Optimization")
    print("=" * 80)
    # Grid search over RRF weights to optimize Recall@50
    weight_configs: list[tuple[str, list[float]]] = [
        ("Equal Weights", [1.0, 1.0, 0.3, 0.3]),
        ("Dense-Heavy (E5 dominant)", [0.5, 1.5, 0.2, 0.4]),
        ("Lexical-Heavy", [1.5, 0.5, 0.4, 0.2]),
        ("Balanced Fusion (Optimal)", [1.0, 1.2, 0.3, 0.4]),
        ("Routed Only (No Global Fallback)", [1.0, 1.2, 0.0, 0.0]),
    ]

    fusion_results: dict[str, dict[str, float]] = {}
    best_config_name = ""
    best_r50 = -1.0
    best_final_rankings: list[list[tuple[str, float]]] = []

    for name, weights in weight_configs:
        fused = reciprocal_rank_fusion(
            [
                best_lexical_fusion,
                routed_e5_rankings,
                global_word_rankings,
                global_e5_rankings,
            ],
            weights=weights,
            c=60.0,
            limit=50,
        )
        metrics = evaluate_rankings(fused, query_ids, relevant, ks=(50,))
        r50 = metrics["recall@50"]
        fusion_results[name] = metrics
        print(f"  {name:<35}: Recall@50 = {r50:.4f}")
        if r50 > best_r50:
            best_r50 = r50
            best_config_name = name
            best_final_rankings = fused

    print(f"\nSelected Fusion Pipeline: '{best_config_name}' with Recall@50 = {best_r50:.4f}")

    # Bootstrap 95% CI on final selected pipeline
    mean_r50, ci_low, ci_high = bootstrap_ci(
        best_final_rankings, query_ids, relevant, k=50, n_boot=1000, seed=42
    )
    print(
        f"Selected Pipeline Recall@50 95% Bootstrap CI: {mean_r50:.4f} [{ci_low:.4f} - {ci_high:.4f}]"
    )

    print("\n" + "=" * 80)
    print("STEP 8: Sliced Evaluation on Canonical Validation Set")
    print("=" * 80)
    train_queries_set = set(train_part["search_query"].astype(str))
    val_q_series = val_contexts["search_query"].astype(str)
    val_params_series = val_contexts["search_infm_params_text"].fillna("").astype(str)
    val_cat_series = val_contexts["search_category"]

    is_seen = pd.Series(val_q_series.isin(list(train_queries_set)), index=val_contexts.index)
    has_filters = pd.Series(val_params_series.str.strip().str.len() > 0, index=val_contexts.index)
    cat_114 = pd.Series(val_cat_series == 114, index=val_contexts.index)
    cat_0 = pd.Series(val_cat_series == 0, index=val_contexts.index)
    query_lens = val_q_series.str.split().str.len()
    is_short = pd.Series(query_lens <= 2, index=val_contexts.index)
    is_long = pd.Series(query_lens >= 4, index=val_contexts.index)

    slices: dict[str, pd.Series] = {
        "seen_query": is_seen,
        "unseen_query": ~is_seen,
        "filters_present": has_filters,
        "filters_absent": ~has_filters,
        "category_114": cat_114,
        "category_0": cat_0,
        "short_query_len_le_2": is_short,
        "long_query_len_ge_4": is_long,
    }
    slice_metrics = evaluate_slices(
        best_final_rankings, query_ids, val_contexts, relevant, slices, ks=(50,)
    )
    for s_name, s_met in slice_metrics.items():
        print(f"  Slice {s_name:<25}: Recall@50 = {s_met.get('recall@50', 0.0):.4f}")

    print("\n" + "=" * 80)
    print("STEP 9: Save Comprehensive Experiment Artifacts")
    print("=" * 80)
    # Save routed lexical metrics
    with open("artifacts/research/routed_lexical_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "global_word": m_global_word,
            "global_char": m_global_char,
            "routed_word": m_routed_word,
            "routed_char": m_routed_char,
            "routed_title_params": m_routed_title_params,
            "routed_filters_params": m_routed_filters,
            "numeric_matching": m_numeric,
            "best_lexical_fusion": m_best_lexical,
        }, f, indent=2)

    # Save routed e5 metrics
    with open("artifacts/research/routed_e5_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "global_e5": m_global_e5,
            "routed_e5": m_routed_e5,
            "routed_e5_fallback": m_routed_e5_fallback,
        }, f, indent=2)

    # Save candidate union metrics
    with open("artifacts/research/candidate_union_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "oracle_lexical": oracle_lexical,
            "oracle_dense": oracle_dense,
            "oracle_combined": oracle_combined,
            "historical_intent": {
                "metrics": m_hist,
                "incremental_coverage_500": delta_cov500,
                "decision": hist_decision,
            },
        }, f, indent=2)

    # Save final ranking metrics
    with open("artifacts/research/final_ranking_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "fusion_grid": fusion_results,
            "selected_config": best_config_name,
            "selected_recall_50": best_r50,
            "bootstrap_ci_95": {"mean": mean_r50, "ci_low": ci_low, "ci_high": ci_high},
            "slices": slice_metrics,
        }, f, indent=2)

    print(f"\nAll experiments complete in {time.time() - started_all:.1f}s. Artifacts saved.")


if __name__ == "__main__":
    main()
