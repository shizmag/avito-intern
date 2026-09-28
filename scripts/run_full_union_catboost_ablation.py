"""Memory-safe apples-to-apples ablation: CatBoost over RRF top-500 vs Full Generator Union."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from avito_candidate_generation.fusion.features import build_pair_features
from avito_candidate_generation.fusion.rrf import reciprocal_rank_fusion


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError, KeyError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError, KeyError):
        return default


def build_features_matrix(
    concat_df: pd.DataFrame,
    source_names: tuple[str, ...] = ("bm25", "dense", "two_tower"),
    floors: dict[str, float] | None = None,
    missing_rank: int = 501,
) -> pd.DataFrame:
    """Build pair features with missing-value defaults."""
    if floors is None:
        floors = {"bm25": -1.0, "dense": -0.255132, "two_tower": -0.734337}

    features = build_pair_features(concat_df, source_names=list(source_names))
    for src in source_names:
        rank_col = f"{src}_rank"
        score_col = f"{src}_score"
        pres_col = f"{src}_present"
        features[pres_col] = features[pres_col].fillna(False).astype("int8")
        features[rank_col] = features[rank_col].fillna(missing_rank).astype("int32")
        floor = floors.get(src, -100.0)
        features[score_col] = features[score_col].fillna(floor).astype("float32")
    features["retriever_count"] = features["retriever_count"].astype("int8")
    return features


def main() -> None:
    t0 = time.time()
    cache_dir = Path("artifacts/real_validation/candidates_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Checking candidate caches and ground truth...")
    bm25_path = cache_dir / "bm25_top500.parquet"
    dense_path = cache_dir / "dense_top500.parquet"
    tt_path = cache_dir / "two_tower_top500.parquet"
    gt_path = Path("artifacts/selected/prepared/validation_ground_truth.parquet")

    if not (
        bm25_path.is_file()
        and dense_path.is_file()
        and tt_path.is_file()
        and gt_path.is_file()
    ):
        print("Missing required cache files!")
        return

    gt_df = pd.read_parquet(gt_path)
    gt_pairs = set(
        zip(
            gt_df["internal_query_id"].astype(str),
            gt_df["item_id"].astype(str),
            strict=True,
        )
    )
    relevant_by_query: dict[str, set[str]] = {}
    for qid, grp in gt_df.groupby("internal_query_id"):
        relevant_by_query[str(qid)] = set(grp["item_id"].astype(str))

    # Read query list
    query_col = pd.read_parquet(bm25_path, columns=["internal_query_id"])[
        "internal_query_id"
    ]
    all_query_ids = query_col.iloc[::500].tolist()
    n_total_queries = len(all_query_ids)
    print(f"Total queries: {n_total_queries}, GT pairs: {len(gt_pairs)}")

    # Split: hash bucket 0 (20%) train, buckets 1..4 (80%) test
    is_train_query = np.array(
        [
            int.from_bytes(hashlib.sha256(f"catboost:{q}".encode()).digest()[:8], "big")
            % 5
            == 0
            for q in all_query_ids
        ]
    )
    train_row_mask = np.repeat(is_train_query, 500)

    n_train_queries = int(is_train_query.sum())
    n_test_queries = n_total_queries - n_train_queries
    print(f"Train queries: {n_train_queries}, Test queries: {n_test_queries}")

    # Step 2: Load train candidates and train both models
    print("\nStep 2: Loading train candidate slices...")
    t_load = time.time()
    bm25_train = pd.read_parquet(bm25_path)[train_row_mask]
    dense_train = pd.read_parquet(dense_path)[train_row_mask]
    tt_train = pd.read_parquet(tt_path)[train_row_mask]
    print(
        f"Loaded train candidates in {time.time() - t_load:.2f}s ({len(bm25_train)} rows per retriever)"
    )

    # Score floors
    floors = {
        "bm25": _safe_float(bm25_train["score"].min() - 1.0, -1.0),
        "dense": _safe_float(dense_train["score"].min() - 1.0, -0.255),
        "two_tower": _safe_float(tt_train["score"].min() - 1.0, -0.734),
    }
    print(f"Computed feature floors: {floors}")

    source_names = ("bm25", "dense", "two_tower")
    feature_cols = [
        "bm25_rank",
        "bm25_score",
        "dense_rank",
        "dense_score",
        "two_tower_rank",
        "two_tower_score",
        "bm25_present",
        "dense_present",
        "two_tower_present",
        "retriever_count",
    ]

    concat_train = cast(
        pd.DataFrame, pd.concat([bm25_train, dense_train, tt_train], ignore_index=True)
    )
    del bm25_train, dense_train, tt_train
    gc.collect()

    print("\n--- Training Model B (Full Generator Union) ---")
    t_tb = time.time()
    features_full_train = build_features_matrix(
        concat_train, source_names, floors=floors
    )
    gt_sub = gt_df[["internal_query_id", "item_id"]].copy()
    gt_sub["label"] = 1
    features_full_train = features_full_train.merge(
        gt_sub, on=["internal_query_id", "item_id"], how="left"
    )
    features_full_train["label"] = features_full_train["label"].fillna(0).astype("int8")
    del gt_sub
    print(
        f"Full union train features: {features_full_train.shape} (positives: {features_full_train['label'].sum()}) in {time.time() - t_tb:.2f}s"
    )

    cb_b = CatBoostClassifier(
        iterations=250,
        depth=7,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=42,
        thread_count=4,
        verbose=False,
    )
    cb_b.fit(features_full_train[feature_cols], features_full_train["label"])
    print(f"Model B fitted in {time.time() - t_tb:.2f}s")

    print("\n--- Training Model A (RRF top-500) ---")
    t_ta = time.time()
    rrf_train = reciprocal_rank_fusion(concat_train, rrf_k=60, limit=500)
    rrf_train_pairs = set(
        zip(
            rrf_train["internal_query_id"].astype(str),
            rrf_train["item_id"].astype(str),
            strict=True,
        )
    )
    del concat_train, rrf_train
    gc.collect()

    is_in_rrf_train = [
        (str(q), str(it)) in rrf_train_pairs
        for q, it in zip(
            features_full_train["internal_query_id"],
            features_full_train["item_id"],
            strict=True,
        )
    ]
    features_rrf_train = features_full_train[is_in_rrf_train].copy()
    del features_full_train, rrf_train_pairs, is_in_rrf_train
    gc.collect()
    print(
        f"RRF top-500 train features: {features_rrf_train.shape} (positives: {features_rrf_train['label'].sum()}) in {time.time() - t_ta:.2f}s"
    )

    cb_a = CatBoostClassifier(
        iterations=250,
        depth=7,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=42,
        thread_count=4,
        verbose=False,
    )
    cb_a.fit(features_rrf_train[feature_cols], features_rrf_train["label"])
    print(f"Model A fitted in {time.time() - t_ta:.2f}s")
    del features_rrf_train
    gc.collect()

    # Step 3: Streamed evaluation on held-out queries (34,629 queries)
    print("\nStep 3: Evaluating on held-out queries in memory-bounded chunks...")
    test_query_indices = np.where(~is_train_query)[0]
    chunk_size = 5000  # 5,000 queries per chunk (~2.5M - 6M candidate rows)

    # Accumulators for Exp A and Exp B
    # Candidate counts
    counts_a: list[int] = []
    counts_b: list[int] = []

    # Coverage: (query_id -> fraction of GT items in candidate pool)
    coverage_scores_a: list[float] = []
    coverage_scores_b: list[float] = []

    # Recall at 10, 20, 50 per query
    recall10_scores_a: list[float] = []
    recall20_scores_a: list[float] = []
    recall50_scores_a: list[float] = []

    recall10_scores_b: list[float] = []
    recall20_scores_b: list[float] = []
    recall50_scores_b: list[float] = []

    # For paired comparison
    paired_diffs_50: list[float] = []

    t_eval_start = time.time()
    for chunk_start in range(0, len(test_query_indices), chunk_size):
        t_chunk = time.time()
        chunk_q_idxs = test_query_indices[chunk_start : chunk_start + chunk_size]

        # Build row mask for this query chunk
        chunk_q_mask = np.zeros(n_total_queries, dtype=bool)
        chunk_q_mask[chunk_q_idxs] = True
        chunk_row_mask = np.repeat(chunk_q_mask, 500)

        # Read candidates for this chunk
        bm25_chunk = pd.read_parquet(bm25_path)[chunk_row_mask]
        dense_chunk = pd.read_parquet(dense_path)[chunk_row_mask]
        tt_chunk = pd.read_parquet(tt_path)[chunk_row_mask]
        concat_chunk = cast(
            pd.DataFrame,
            pd.concat([bm25_chunk, dense_chunk, tt_chunk], ignore_index=True),
        )
        del bm25_chunk, dense_chunk, tt_chunk
        gc.collect()

        # Build Full Union features for chunk
        features_chunk = build_features_matrix(
            concat_chunk, source_names, floors=floors
        )

        # RRF top-500 candidate selection for chunk
        rrf_chunk = reciprocal_rank_fusion(concat_chunk, rrf_k=60, limit=500)
        rrf_chunk_pairs = set(
            zip(
                rrf_chunk["internal_query_id"].astype(str),
                rrf_chunk["item_id"].astype(str),
                strict=True,
            )
        )
        del concat_chunk, rrf_chunk
        gc.collect()

        # Filter for Exp A
        is_rrf = [
            (str(q), str(it)) in rrf_chunk_pairs
            for q, it in zip(
                features_chunk["internal_query_id"],
                features_chunk["item_id"],
                strict=True,
            )
        ]
        features_chunk_a = features_chunk[is_rrf].copy()
        del rrf_chunk_pairs, is_rrf
        gc.collect()

        # Predict Exp A
        features_chunk_a["score"] = cb_a.predict_proba(features_chunk_a[feature_cols])[
            :, 1
        ]
        features_chunk_a = features_chunk_a.sort_values(
            ["internal_query_id", "score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )  # pyright: ignore[reportCallIssue]

        # Predict Exp B
        features_chunk["score"] = cb_b.predict_proba(features_chunk[feature_cols])[:, 1]
        features_chunk = features_chunk.sort_values(
            ["internal_query_id", "score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )  # pyright: ignore[reportCallIssue]

        # Group and evaluate per query in chunk
        grouped_a = dict(tuple(features_chunk_a.groupby("internal_query_id")))
        grouped_b = dict(tuple(features_chunk.groupby("internal_query_id")))
        del features_chunk_a, features_chunk
        gc.collect()

        for qid in [all_query_ids[i] for i in chunk_q_idxs]:
            gt_items = relevant_by_query.get(qid, set())
            n_gt = len(gt_items)

            # Exp A evaluation
            df_qa = grouped_a.get(qid)
            if df_qa is not None and not df_qa.empty:
                c_items_a = df_qa["item_id"].astype(str).tolist()
                counts_a.append(len(c_items_a))
                if n_gt > 0:
                    cov_a = len(set(c_items_a) & gt_items) / n_gt
                    r10_a = len(set(c_items_a[:10]) & gt_items) / n_gt
                    r20_a = len(set(c_items_a[:20]) & gt_items) / n_gt
                    r50_a = len(set(c_items_a[:50]) & gt_items) / n_gt
                else:
                    cov_a, r10_a, r20_a, r50_a = 0.0, 0.0, 0.0, 0.0
            else:
                counts_a.append(0)
                cov_a, r10_a, r20_a, r50_a = 0.0, 0.0, 0.0, 0.0

            # Exp B evaluation
            df_qb = grouped_b.get(qid)
            if df_qb is not None and not df_qb.empty:
                c_items_b = df_qb["item_id"].astype(str).tolist()
                counts_b.append(len(c_items_b))
                if n_gt > 0:
                    cov_b = len(set(c_items_b) & gt_items) / n_gt
                    r10_b = len(set(c_items_b[:10]) & gt_items) / n_gt
                    r20_b = len(set(c_items_b[:20]) & gt_items) / n_gt
                    r50_b = len(set(c_items_b[:50]) & gt_items) / n_gt
                else:
                    cov_b, r10_b, r20_b, r50_b = 0.0, 0.0, 0.0, 0.0
            else:
                counts_b.append(0)
                cov_b, r10_b, r20_b, r50_b = 0.0, 0.0, 0.0, 0.0

            if n_gt > 0:
                coverage_scores_a.append(cov_a)
                recall10_scores_a.append(r10_a)
                recall20_scores_a.append(r20_a)
                recall50_scores_a.append(r50_a)

                coverage_scores_b.append(cov_b)
                recall10_scores_b.append(r10_b)
                recall20_scores_b.append(r20_b)
                recall50_scores_b.append(r50_b)

                paired_diffs_50.append(r50_b - r50_a)

        del grouped_a, grouped_b
        gc.collect()

        processed_so_far = min(chunk_start + chunk_size, len(test_query_indices))
        print(
            f"Evaluated queries {processed_so_far}/{len(test_query_indices)} in {time.time() - t_chunk:.2f}s "
            f"(current mean R@50: Exp A={np.mean(recall50_scores_a):.4f}, Exp B={np.mean(recall50_scores_b):.4f})"
        )

    print(f"\nAll held-out queries evaluated in {time.time() - t_eval_start:.2f}s!")

    # Summary Statistics
    counts_a_arr = np.array(counts_a)
    counts_b_arr = np.array(counts_b)

    mean_cnt_a = _safe_float(np.mean(counts_a_arr))
    p50_cnt_a = _safe_float(np.median(counts_a_arr))
    p95_cnt_a = _safe_float(np.percentile(counts_a_arr, 95))
    max_cnt_a = _safe_int(np.max(counts_a_arr))

    mean_cnt_b = _safe_float(np.mean(counts_b_arr))
    p50_cnt_b = _safe_float(np.median(counts_b_arr))
    p95_cnt_b = _safe_float(np.percentile(counts_b_arr, 95))
    max_cnt_b = _safe_int(np.max(counts_b_arr))

    cov_a = _safe_float(np.mean(coverage_scores_a))
    r10_a = _safe_float(np.mean(recall10_scores_a))
    r20_a = _safe_float(np.mean(recall20_scores_a))
    r50_a = _safe_float(np.mean(recall50_scores_a))

    cov_b = _safe_float(np.mean(coverage_scores_b))
    r10_b = _safe_float(np.mean(recall10_scores_b))
    r20_b = _safe_float(np.mean(recall20_scores_b))
    r50_b = _safe_float(np.mean(recall50_scores_b))

    delta_50 = r50_b - r50_a
    delta_50_pp = delta_50 * 100.0

    # Paired Bootstrap CI
    rng = np.random.default_rng(42)
    diffs_arr = np.array(paired_diffs_50, dtype=np.float64)
    boot_means: list[float] = []
    n_diffs = len(diffs_arr)
    for _ in range(1000):
        sample = rng.choice(diffs_arr, size=n_diffs, replace=True)
        boot_means.append(_safe_float(np.mean(sample)))

    ci_lower = _safe_float(np.percentile(boot_means, 2.5))
    ci_upper = _safe_float(np.percentile(boot_means, 97.5))

    decision = "full_union" if delta_50 >= 0.005 else "rrf_top500"

    print("\n========================================================")
    print("FINAL ABLATION RESULTS")
    print("========================================================")
    print("Experiment A: CatBoost over RRF top-500")
    print(
        f"  Candidates: mean={mean_cnt_a:.1f}, p50={p50_cnt_a:.1f}, p95={p95_cnt_a:.1f}, max={max_cnt_a}"
    )
    print(f"  Candidate pool coverage: {cov_a:.6f} ({cov_a * 100:.2f}%)")
    print(f"  Recall@10: {r10_a:.6f}")
    print(f"  Recall@20: {r20_a:.6f}")
    print(f"  Recall@50: {r50_a:.6f}")
    print()
    print("Experiment B: CatBoost over Full Generator Union")
    print(
        f"  Candidates: mean={mean_cnt_b:.1f}, p50={p50_cnt_b:.1f}, p95={p95_cnt_b:.1f}, max={max_cnt_b}"
    )
    print(f"  Candidate pool coverage: {cov_b:.6f} ({cov_b * 100:.2f}%)")
    print(f"  Recall@10: {r10_b:.6f}")
    print(f"  Recall@20: {r20_b:.6f}")
    print(f"  Recall@50: {r50_b:.6f}")
    print()
    print("Comparison:")
    print(f"  Delta Recall@50: {delta_50:+.6f} ({delta_50_pp:+.4f} percentage points)")
    print(f"  Paired 95% Bootstrap CI: [{ci_lower:+.6f}, {ci_upper:+.6f}]")
    print("  Decision Rule: Delta >= +0.005 (+0.5 pp) -> full_union, else rrf_top500")
    print(f"  Selected Candidate Pool Architecture: >>> {decision.upper()} <<<")
    print("========================================================")

    result_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "split": "cold-item-validation",
        "protocol": "disjoint hash bucket 0 (20%) trains selector; buckets 1-4 (80%) held out",
        "n_train_queries": n_train_queries,
        "n_test_queries": n_test_queries,
        "candidate_k_per_retriever": 500,
        "experiment_a_rrf_top500": {
            "name": "CatBoost over RRF top-500",
            "candidate_count_mean": mean_cnt_a,
            "candidate_count_p50": p50_cnt_a,
            "candidate_count_p95": p95_cnt_a,
            "candidate_count_max": max_cnt_a,
            "candidate_pool_coverage": cov_a,
            "recall@10": r10_a,
            "recall@20": r20_a,
            "recall@50": r50_a,
        },
        "experiment_b_full_union": {
            "name": "CatBoost over Full Generator Union",
            "candidate_count_mean": mean_cnt_b,
            "candidate_count_p50": p50_cnt_b,
            "candidate_count_p95": p95_cnt_b,
            "candidate_count_max": max_cnt_b,
            "candidate_pool_coverage": cov_b,
            "recall@10": r10_b,
            "recall@20": r20_b,
            "recall@50": r50_b,
        },
        "comparison": {
            "delta_recall@50": delta_50,
            "delta_recall@50_percentage_points": delta_50_pp,
            "paired_bootstrap_95_ci": [ci_lower, ci_upper],
            "decision_threshold_pp": 0.5,
            "decision": decision,
        },
        "features": feature_cols,
        "parameters": {
            "iterations": 250,
            "depth": 7,
            "learning_rate": 0.08,
            "random_seed": 42,
        },
        "total_runtime_sec": round(time.time() - t0, 2),
    }

    out_file = Path("artifacts/real_validation/catboost_candidate_pool_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved structured results to {out_file}")


if __name__ == "__main__":
    main()
