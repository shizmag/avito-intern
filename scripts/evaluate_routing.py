"""Evaluate microcategory routing predictors on canonical validation set."""

import time
import json
import pandas as pd
from avito_candidate_generation.research import context_frame, context_folds, CONTEXT_COLUMNS
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
    evaluate_microcategory_routing,
)

def main():
    print("Loading data...")
    t0 = time.time()
    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids = set(benchmark_items["item_id"])
    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    # Train set: folds != 0 (80%)
    train_part = merged[merged["fold"] != 0].copy()
    # Val set: fold == 0 (20%) with known benchmark positive
    val_rows = merged[merged["fold"] == 0]
    val_with_bm_pos = val_rows[val_rows["item_id"].isin(benchmark_ids)].copy()

    val_n_q = val_with_bm_pos["internal_query_id"].nunique()
    print(f"Data loaded in {time.time()-t0:.1f}s. Train rows: {len(train_part):,}, Val contexts: {val_n_q:,}")

    # Ground truth map: internal_query_id -> set of positive item_ids
    relevant = val_with_bm_pos.groupby("internal_query_id")["item_id"].apply(set).to_dict()
    val_contexts = val_with_bm_pos[[*CONTEXT_COLUMNS, "internal_query_id"]].drop_duplicates().reset_index(drop=True)
    query_ids = val_contexts["internal_query_id"].astype(str).tolist()
    query_texts = val_contexts["search_query"].astype(str).tolist()

    item_to_microcat = dict(zip(benchmark_items["item_id"].astype(str), benchmark_items["item_microcat_id"].astype(str)))
    seen_queries = set(train_part["search_query"].astype(str))
    val_seen = sum(1 for q in query_texts if q in seen_queries)
    print(f"Validation queries seen in train: {val_seen}/{len(query_texts)} ({val_seen/len(query_texts):.1%})")

    # 1. Exact posterior predictor
    print("\n--- Fitting Exact Posterior ---")
    t0 = time.time()
    exact = ExactPosteriorPredictor()
    exact.fit(train_part, query_col="search_query", category_col="item_microcat_id")
    print(f"Exact posterior fit in {time.time()-t0:.1f}s")

    exact_preds = {k: exact.predict_top_k(query_texts, k=k) for k in [1, 3, 5, 10]}
    exact_metrics = evaluate_microcategory_routing(
        exact_preds, query_ids, query_texts, relevant, item_to_microcat, seen_queries, len(benchmark_items)
    )

    # 2. Generalizing Classifier Predictor
    print("\n--- Fitting Generalizing Classifier (TF-IDF + SGDClassifier) ---")
    t0 = time.time()
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42)
    # Fit on training set
    clf.fit(train_part)
    print(f"Generalizing classifier fit in {time.time()-t0:.1f}s")

    clf_preds = {k: clf.predict_top_k(val_contexts, k=k) for k in [1, 3, 5, 10]}
    clf_metrics = evaluate_microcategory_routing(
        clf_preds, query_ids, query_texts, relevant, item_to_microcat, seen_queries, len(benchmark_items)
    )

    # 3. Hybrid Predictor
    print("\n--- Fitting Hybrid Predictor (Exact + Generalizing Fallback) ---")
    t0 = time.time()
    hybrid = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=clf,
        min_count=1,
        blend_strategy="fallback",
    )
    print(f"Hybrid predictor ready in {time.time()-t0:.1f}s")

    hybrid_preds = {k: hybrid.predict_top_k(val_contexts, k=k) for k in [1, 3, 5, 10]}
    hybrid_metrics = evaluate_microcategory_routing(
        hybrid_preds, query_ids, query_texts, relevant, item_to_microcat, seen_queries, len(benchmark_items)
    )

    print("\n" + "="*80)
    print("ROUTING QUALITY COMPARISON (Validation queries = 5,347)")
    print("="*80)
    print(f"{'Model':<22} | {'Top-1 Cont':<10} | {'Top-3 Cont':<10} | {'Top-5 Cont':<10} | {'Top-10 Cont':<10}")
    print("-" * 80)
    for name, m in [("Exact Posterior", exact_metrics), ("TF-IDF Classifier", clf_metrics), ("Hybrid Model", hybrid_metrics)]:
        print(f"{name:<22} | {m[1]['overall']['positive_item_containment']:<10.4f} | {m[3]['overall']['positive_item_containment']:<10.4f} | {m[5]['overall']['positive_item_containment']:<10.4f} | {m[10]['overall']['positive_item_containment']:<10.4f}")

    print("\nUNSEEN QUERY CONTAINMENT (N = %d):" % (len(query_texts) - val_seen))
    print(f"{'Model':<22} | {'Top-1 Cont':<10} | {'Top-3 Cont':<10} | {'Top-5 Cont':<10} | {'Top-10 Cont':<10}")
    print("-" * 80)
    for name, m in [("Exact Posterior", exact_metrics), ("TF-IDF Classifier", clf_metrics), ("Hybrid Model", hybrid_metrics)]:
        print(f"{name:<22} | {m[1]['unseen_query']['positive_item_containment']:<10.4f} | {m[3]['unseen_query']['positive_item_containment']:<10.4f} | {m[5]['unseen_query']['positive_item_containment']:<10.4f} | {m[10]['unseen_query']['positive_item_containment']:<10.4f}")

    print("\nSEEN QUERY CONTAINMENT (N = %d):" % val_seen)
    print(f"{'Model':<22} | {'Top-1 Cont':<10} | {'Top-3 Cont':<10} | {'Top-5 Cont':<10} | {'Top-10 Cont':<10}")
    print("-" * 80)
    for name, m in [("Exact Posterior", exact_metrics), ("TF-IDF Classifier", clf_metrics), ("Hybrid Model", hybrid_metrics)]:
        print(f"{name:<22} | {m[1]['seen_query']['positive_item_containment']:<10.4f} | {m[3]['seen_query']['positive_item_containment']:<10.4f} | {m[5]['seen_query']['positive_item_containment']:<10.4f} | {m[10]['seen_query']['positive_item_containment']:<10.4f}")

    print("\nHYBRID MODEL DETAILED METRICS:")
    print(f"{'K':<5} | {'Hit Recall':<12} | {'Containment':<12} | {'Mean Corpus':<12} | {'Median':<8} | {'P95':<8} | {'Max':<8}")
    print("-" * 80)
    for k in [1, 3, 5, 10]:
        stats = hybrid_metrics[k]['overall']
        print(f"Top-{k:<2} | {stats['category_hit_recall']:<12.4f} | {stats['positive_item_containment']:<12.4f} | {stats['corpus_size_mean']:<12.1f} | {stats['corpus_size_median']:<8.0f} | {stats['corpus_size_p95']:<8.0f} | {stats['corpus_size_max']:<8.0f}")

    # Save to artifacts/research/microcategory_metrics.json
    import os
    os.makedirs("artifacts/research", exist_ok=True)
    with open("artifacts/research/microcategory_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "exact_posterior": exact_metrics,
            "classifier": clf_metrics,
            "hybrid": hybrid_metrics,
            "queries_evaluated": len(query_texts),
            "seen_queries": val_seen,
            "unseen_queries": len(query_texts) - val_seen,
        }, f, indent=2, ensure_ascii=False)
    print("\nSaved artifacts/research/microcategory_metrics.json successfully.")

if __name__ == "__main__":
    main()
