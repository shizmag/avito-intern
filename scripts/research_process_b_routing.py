"""Process B: Microcategory Routing Experiments (H4, H5).

Evaluates:
- Routing Top-K for K in [5, 8, 10, 12]
- Containment overall and on critical slices: seen_query vs unseen_query
- Corpus size statistics (mean, median/p50, p95, min, max)
- Routing confidence and distribution
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.research import (
    context_folds,
    context_frame,
    ground_truth_map,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def evaluate_routing_k(
    hybrid: HybridMicrocategoryPredictor,
    val_contexts: pd.DataFrame,
    relevant: dict[str, set[str]],
    item_microcat_map: dict[str, str],
    microcat_to_items: dict[str, set[str]],
    k_values: list[int],
    train_queries_set: set[str],
) -> dict[str, Any]:
    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    max_k = max(k_values)
    all_preds = hybrid.predict_top_k(val_contexts, k=max_k)

    # Pre-extract slice masks
    val_queries_norm = (
        val_contexts["search_query"].fillna("").astype(str).str.lower().str.strip()
    )
    is_seen = val_queries_norm.isin(train_queries_set).to_numpy()
    is_unseen = ~is_seen
    is_cat0 = (val_contexts["search_category"] == 0).to_numpy()

    # Pre-extract ground truth microcats per query
    gt_microcats: list[set[str]] = []
    for qid in query_ids:
        pos_items = relevant.get(qid, set())
        cats = {str(item_microcat_map[it]) for it in pos_items if it in item_microcat_map}
        gt_microcats.append(cats)

    results: dict[str, Any] = {}

    for k in k_values:
        containment_all: list[bool] = []
        corpus_sizes: list[int] = []

        for idx, (pred_cats_all, gt_cats) in enumerate(zip(all_preds, gt_microcats, strict=True)):
            topk_cats = {str(c) for c in pred_cats_all[:k]}
            hit = bool(topk_cats & gt_cats) if gt_cats else False
            containment_all.append(hit)

            c_size = sum(len(microcat_to_items.get(str(c), set())) for c in topk_cats)
            corpus_sizes.append(c_size)

        cont_arr = np.array(containment_all, dtype=bool)
        c_sizes = np.array(corpus_sizes, dtype=int)

        res_k = {
            "k": k,
            "containment_overall": float(cont_arr.mean()),
            "containment_seen": float(cont_arr[is_seen].mean()) if is_seen.any() else 0.0,
            "containment_unseen": float(cont_arr[is_unseen].mean()) if is_unseen.any() else 0.0,
            "containment_cat0": float(cont_arr[is_cat0].mean()) if is_cat0.any() else 0.0,
            "corpus_size_mean": float(c_sizes.mean()),
            "corpus_size_p50": float(np.median(c_sizes)),
            "corpus_size_p95": float(np.percentile(c_sizes, 95)),
            "corpus_size_min": int(c_sizes.min()),
            "corpus_size_max": int(c_sizes.max()),
        }
        results[f"K={k}"] = res_k
        print(
            f"K={k:<2}: Containment={res_k['containment_overall']:.4f} "
            f"(seen={res_k['containment_seen']:.4f}, unseen={res_k['containment_unseen']:.4f}) | "
            f"Corpus size: mean={res_k['corpus_size_mean']:.1f}, p50={res_k['corpus_size_p50']:.0f}, p95={res_k['corpus_size_p95']:.0f}"
        )

    return results


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("PROCESS B: ROUTING EXPERIMENTS (H4: Top-K Widening, H5: Generalization)")
    print("=" * 70)

    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
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

    train_queries_set = set(
        train_part["search_query"].dropna().astype(str).str.lower().str.strip()
    )

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"].astype(str)))

    microcat_to_items: dict[str, set[str]] = {}
    for item_id, mc in item_microcat_map.items():
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    print("Fitting hybrid microcategory routing models...")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")

    print("\nEvaluating routing for K in [5, 8, 10, 12]...")
    k_results = evaluate_routing_k(
        hybrid=hybrid,
        val_contexts=val_contexts,
        relevant=relevant,
        item_microcat_map=item_microcat_map,
        microcat_to_items=microcat_to_items,
        k_values=[5, 8, 10, 12],
        train_queries_set=train_queries_set,
    )

    # Save artifact
    out_path = Path("artifacts/research_v3/routing_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(k_results, f, indent=2)

    runtime_sec = time.time() - t0
    print(f"\nProcess B completed in {runtime_sec:.1f}s. Artifact saved to {out_path}")


if __name__ == "__main__":
    main()
