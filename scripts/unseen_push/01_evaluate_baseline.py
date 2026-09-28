"""Evaluate current champion pipeline baseline on the strict unseen split.

Produces:
1. Routing containment (@1, @3, @5, @10) overall, seen, unseen.
2. Candidate pool union coverage (@50, @100, @200, @500, @1000) overall, seen, unseen.
3. Final Recall@50, @100, @200, @500 overall, seen, unseen.
4. Benchmark-weighted proxy: 0.35 * Recall_seen@50 + 0.65 * Recall_unseen@50.
5. Saves artifacts/unseen_push/metrics/baseline_metrics.json.
6. Saves artifacts/unseen_push/candidates/baseline_candidates_val.parquet (top-1000 candidates with scores/ranks).
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
    reciprocal_rank_fusion,
)
from avito_candidate_generation.research import field_text
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


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STEP 1: EVALUATE BASELINE ON STRICT UNSEEN SPLIT")
    print("=" * 80)

    val_dir = Path("artifacts/unseen_push/validation")
    metrics_dir = Path("artifacts/unseen_push/metrics")
    cand_dir = Path("artifacts/unseen_push/candidates")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    with (val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
        relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

    with (val_dir / "loc_coords.json").open("r", encoding="utf-8") as f:
        loc_coords = {int(k): v for k, v in json.load(f).items()}

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    is_seen = val_contexts["is_seen"].to_numpy()
    is_unseen = val_contexts["is_unseen"].to_numpy()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(
        zip(item_id_list, benchmark_items["item_microcat_id"].astype(str))
    )
    item_lat_map = dict(
        zip(
            item_id_list,
            benchmark_items["item_latitude"].astype(float).fillna(0.0),
        )
    )
    item_lon_map = dict(
        zip(
            item_id_list,
            benchmark_items["item_longitude"].astype(float).fillna(0.0),
        )
    )
    item_titles = [
        str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()
    ]

    microcat_to_items: dict[str, set[str]] = {}
    for it, mc in item_microcat_map.items():
        microcat_to_items.setdefault(mc, set()).add(it)

    print(
        f"Loaded data in {time.time() - t0:.1f}s. Validation queries: {len(val_contexts)} (Seen: {is_seen.sum()}, Unseen: {is_unseen.sum()})"
    )

    # 1. Routing Evaluation
    print("\n--- [1/5] Fitting Baseline Router (Hybrid K=10) ---")
    t_start = time.time()
    exact = ExactPosteriorPredictor().fit(
        train_part, query_col="search_query", category_col="item_microcat_id"
    )
    clf = GeneralizingClassifierPredictor(
        min_df=5, max_features=50000, alpha=1e-5, seed=42
    ).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=clf,
        min_count=1,
        blend_strategy="fallback",
    )

    preds_k10 = hybrid.predict_top_k(val_contexts, k=10)
    print(f"Router fitted and predicted in {time.time() - t_start:.1f}s")

    # Routing containment evaluation
    routing_metrics: dict[str, Any] = {}
    for k in [1, 3, 5, 10]:
        containment_list = []
        corpus_sizes = []
        for qid, preds in zip(query_ids, preds_k10, strict=True):
            topk_cats = set(preds[:k])
            gt_items = relevant.get(qid, set())
            gt_cats = {item_microcat_map[it] for it in gt_items if it in item_microcat_map}
            hit = bool(topk_cats & gt_cats) if gt_cats else False
            containment_list.append(hit)
            c_size = sum(len(microcat_to_items.get(c, set())) for c in topk_cats)
            corpus_sizes.append(c_size)

        c_arr = np.array(containment_list, dtype=bool)
        c_sizes = np.array(corpus_sizes, dtype=int)
        routing_metrics[f"top_{k}"] = {
            "containment_overall": float(c_arr.mean()),
            "containment_seen": float(c_arr[is_seen].mean()),
            "containment_unseen": float(c_arr[is_unseen].mean()),
            "corpus_mean": float(c_sizes.mean()),
            "corpus_p95": float(np.percentile(c_sizes, 95)),
        }
        print(
            f"  Routing Top-{k:<2}: Overall={c_arr.mean():.4f}, Seen={c_arr[is_seen].mean():.4f}, UNSEEN={c_arr[is_unseen].mean():.4f}"
        )

    # Allowed items for K=10
    allowed_items_per_query: list[set[str]] = []
    for cats in preds_k10:
        allowed: set[str] = set()
        for c in cats:
            allowed.update(microcat_to_items.get(c, set()))
        allowed_items_per_query.append(allowed)

    # Free memory
    del train_part, exact, clf
    gc.collect()

    # 2. Sparse Retrieval
    print("\n--- [2/5] Running Sparse Retrieval (depth=1000) ---")
    t_start = time.time()
    field_index = FieldAwareSparseIndex.fit(
        benchmark_items,
        branch_b_analyzer="char_wb",
        branch_b_ngram_range=(3, 5),
        min_df=2,
    )
    title_char_index = SparseBranchIndex.fit(
        item_id_list,
        item_titles,
        name="title_char",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )

    query_title_texts = field_text(val_contexts, ("search_query",))
    query_title_params_texts = field_text(
        val_contexts, ("search_query", "search_infm_params_text")
    )

    routed_word = field_index.branches["branch_a"].retrieve_routed(
        query_title_texts, allowed_items_per_query, k=1000
    )
    routed_char = title_char_index.retrieve_routed(
        query_title_texts, allowed_items_per_query, k=1000
    )
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(
        query_title_params_texts, allowed_items_per_query, k=1000
    )
    global_word = field_index.branches["branch_a"].retrieve(query_title_texts, k=300)
    global_char = title_char_index.retrieve(query_title_texts, k=300)

    del field_index, title_char_index
    gc.collect()
    print(f"Sparse retrieval done in {time.time() - t_start:.1f}s")

    # 3. Dense E5 Retrieval
    print("\n--- [3/5] Running Dense E5 Retrieval (depth=1000) ---")
    t_start = time.time()
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_items_json = [
        str(x)
        for x in json.loads(
            Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text()
        )
    ]
    e5_item_embeddings = np.load(
        "artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r"
    )
    e5_retriever = RoutedDenseE5Retriever(
        item_ids=e5_items_json, item_embeddings=e5_item_embeddings
    )

    routed_e5 = e5_retriever.retrieve_routed(
        val_query_embeddings, allowed_items_per_query, k=1000
    )
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=300)

    del e5_retriever
    gc.collect()
    print(f"Dense E5 retrieval done in {time.time() - t_start:.1f}s")

    # 4. Candidate Fusion & Candidate Pool Coverage
    print("\n--- [4/5] Candidate Fusion & Candidate Pool Coverage ---")
    sources = [
        routed_e5,
        routed_char,
        routed_word,
        routed_title_params,
        global_e5,
        global_word,
        global_char,
    ]
    weights = [1.5, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3]
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    # Candidate pool coverage at 50, 100, 200, 500, 1000
    pool_coverage: dict[str, dict[str, float]] = {}
    for depth in [50, 100, 200, 500, 1000]:
        cov_all: list[float] = []
        for qid, ranking in zip(query_ids, raw_rrf, strict=True):
            cand_set = {it for it, _ in ranking[:depth]}
            gt_set = relevant.get(qid, set())
            cov = len(cand_set & gt_set) / max(len(gt_set), 1)
            cov_all.append(cov)
        cov_arr = np.array(cov_all)
        pool_coverage[f"coverage@{depth}"] = {
            "overall": float(cov_arr.mean()),
            "seen": float(cov_arr[is_seen].mean()),
            "unseen": float(cov_arr[is_unseen].mean()),
        }
        print(
            f"  Candidate Coverage@{depth:<4}: Overall={cov_arr.mean():.4f}, Seen={cov_arr[is_seen].mean():.4f}, UNSEEN={cov_arr[is_unseen].mean():.4f}"
        )

    # 5. Local-First Geo-Cascade Ranking Baseline
    print("\n--- [5/5] Local-First Geo-Cascade Ranking ---")
    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))
    query_deliv_map = dict(zip(query_ids, val_contexts["search_is_delivery_search"]))

    same_loc_bonus = 30.0
    dist_decay = 0.25

    final_rankings: list[list[tuple[str, float]]] = []
    cand_records: list[dict[str, Any]] = []

    for q_idx, (qid, ranking) in enumerate(zip(query_ids, raw_rrf, strict=True)):
        q_loc = query_loc_map[qid]
        q_cat = query_cat_map[qid]
        q_coord = loc_coords.get(int(q_loc)) if pd.notna(q_loc) else None
        q_lat = q_coord["lat"] if q_coord else None
        q_lon = q_coord["lon"] if q_coord else None
        pos_set = relevant.get(qid, set())

        # Microcategory predictions for this query
        q_pred_mc = preds_k10[q_idx]
        max_rrf = ranking[0][1] if ranking else 1.0

        boosted: list[tuple[str, float]] = []

        for rrf_rank, (item_id, score) in enumerate(ranking):
            it_loc = item_loc_map.get(item_id)
            it_cat = item_cat_map.get(item_id)
            it_mc = item_microcat_map.get(item_id, "")
            it_lat = item_lat_map.get(item_id, 0.0)
            it_lon = item_lon_map.get(item_id, 0.0)

            # Category match multiplier
            if q_cat == 0:
                cat_mult = 1.0
            else:
                cat_mult = 1.0 if (it_cat == q_cat) else 0.001

            # Geo match multiplier & distance
            dist_km = 500.0
            geo_tier = 4
            if it_loc == q_loc:
                geo_mult = same_loc_bonus
                geo_tier = 1
                dist_km = 0.0
            elif (
                q_lat is not None
                and q_lon is not None
                and it_lat != 0.0
                and it_lon != 0.0
            ):
                dist_km = float(haversine_km(q_lat, q_lon, it_lat, it_lon))
                geo_mult = float(np.exp(-dist_decay * (dist_km / 10.0)))
                geo_tier = 2 if dist_km <= 50.0 else 3
            else:
                geo_mult = 0.05
                geo_tier = 4

            final_score = score * cat_mult * geo_mult
            boosted.append((item_id, final_score))

            # Store top-1000 candidate records for Worker C features
            try:
                mc_rank = q_pred_mc.index(it_mc) + 1
            except ValueError:
                mc_rank = -1

            cand_records.append(
                {
                    "qid": qid,
                    "item_id": item_id,
                    "label": int(item_id in pos_set),
                    "is_unseen": int(is_unseen[q_idx]),
                    "rrf_rank": rrf_rank + 1,
                    "rrf_score": score,
                    "norm_rrf_score": score / (max_rrf + 1e-6),
                    "inv_rrf_rank": 1.0 / (rrf_rank + 2.0),
                    "geo_tier": geo_tier,
                    "same_loc": int(it_loc == q_loc),
                    "dist_km": dist_km,
                    "log1p_dist": float(np.log1p(dist_km)),
                    "valid_cat": int(it_cat == q_cat or q_cat == 0),
                    "is_deliv": int(query_deliv_map[qid] or 0),
                    "mc_rank": mc_rank,
                    "cascade_score": final_score,
                }
            )

        boosted.sort(key=lambda p: (-p[1], p[0]))
        final_rankings.append(boosted[:50])

    # Evaluate final Recall
    final_metrics: dict[str, Any] = {}
    for k in [50, 100, 200, 500]:
        rec_all: list[float] = []
        for qid, ranking in zip(query_ids, final_rankings, strict=True):
            top_ids = {it for it, _ in ranking[:k]}
            gt = relevant.get(qid, set())
            rec_all.append(len(top_ids & gt) / max(len(gt), 1))
        rec_arr = np.array(rec_all)
        final_metrics[f"recall@{k}"] = {
            "overall": float(rec_arr.mean()),
            "seen": float(rec_arr[is_seen].mean()),
            "unseen": float(rec_arr[is_unseen].mean()),
        }
        print(
            f"  Final Recall@{k:<4}: Overall={rec_arr.mean():.4f}, Seen={rec_arr[is_seen].mean():.4f}, UNSEEN={rec_arr[is_unseen].mean():.4f}"
        )

    r_seen50 = final_metrics["recall@50"]["seen"]
    r_unseen50 = final_metrics["recall@50"]["unseen"]
    benchmark_proxy = 0.35 * r_seen50 + 0.65 * r_unseen50
    print("\n" + "=" * 80)
    print(f"BASELINE SUMMARY:")
    print(f"  Official Previous:     0.646469")
    print(f"  Official Last:         0.6451")
    print(f"  Offline Overall R@50:  {final_metrics['recall@50']['overall']:.4f}")
    print(f"  Offline Seen R@50:     {r_seen50:.4f}")
    print(f"  Offline UNSEEN R@50:   {r_unseen50:.4f}")
    print(f"  Benchmark Proxy Score: {benchmark_proxy:.4f}")
    print("=" * 80)

    # Save metrics JSON
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "official_recall_50": 0.645100,
        "baseline_overall_r50": final_metrics["recall@50"]["overall"],
        "baseline_seen_r50": r_seen50,
        "baseline_unseen_r50": r_unseen50,
        "benchmark_proxy": benchmark_proxy,
        "routing_metrics": routing_metrics,
        "candidate_pool_coverage": pool_coverage,
        "final_recall_metrics": final_metrics,
    }
    with (metrics_dir / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save candidate pool for Worker C (features and ranker)
    print("\nSaving baseline candidates DataFrame for downstream workers...")
    t_start = time.time()
    cand_df = pd.DataFrame(cand_records)
    cand_df.to_parquet(
        cand_dir / "baseline_candidates_val.parquet",
        index=False,
        compression="snappy",
    )
    print(
        f"Saved {len(cand_df):,} candidate rows to {cand_dir / 'baseline_candidates_val.parquet'} in {time.time() - t_start:.1f}s"
    )
    print(f"\nBaseline evaluation completed successfully in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
