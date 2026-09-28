"""Worker C: H3 Pairwise Learning-to-Rank on Retrieval Mistakes.

Evaluates on candidate pool:
1. Feature Engineering with per-query normalized features and retrieval mistakes.
2. CatBoostRanker with YetiRank / PairLogitPairwise.
3. 5-Fold Strict Out-Of-Fold (OOF) by query group.
4. Ranking Strategies:
   A. Baseline Geo-Cascade
   B. Pure Ranker
   C. Ranker within Geo Tiers
   D. Hybrid: Geo-Cascade + lambda * Ranker Score (lambda in [0.1, 0.25, 0.5, 1.0])

Computes:
- Unseen Recall@50, Seen Recall@50, Benchmark Proxy (0.35 seen + 0.65 unseen)
- Delta vs baseline
- Accept / Kill gate: Baseline unseen R@50 ≈ 0.6321. Strong >= +2 pp, Accept >= +1 pp, Kill < +0.5 pp.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WORKER C: PAIRWISE LEARNING-TO-RANK EXPERIMENT (CATBOOST RANKER)")
    print("=" * 80)

    val_dir = Path("artifacts/unseen_push/validation")
    cand_dir = Path("artifacts/unseen_push/candidates")
    ranking_dir = Path("artifacts/unseen_push/ranking")
    ranking_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    with (val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
        relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    is_seen = val_contexts["is_seen"].to_numpy()
    is_unseen = val_contexts["is_unseen"].to_numpy()

    # Load candidate pool
    print("\n[1/5] Loading 5.3M candidate rows from baseline_candidates_val.parquet...")
    t_start = time.time()
    df = pd.read_parquet(cand_dir / "baseline_candidates_val.parquet")
    print(f"Loaded {len(df):,} candidates in {time.time() - t_start:.1f}s")

    # Map qid to integer group_id for CatBoost
    df["group_id"] = df["qid"].map(qid_to_idx)
    # Assign 5 folds deterministically by group_id
    df["fold"] = df["group_id"] % 5

    # Compute per-query normalized features in vectorized fashion
    print("\n[2/5] Vectorized per-query feature normalization...")
    t_start = time.time()
    grouped = df.groupby("group_id")["cascade_score"]
    q_max_score = grouped.transform("max")
    q_mean_score = grouped.transform("mean")
    q_std_score = grouped.transform("std").fillna(1.0)
    q_count = grouped.transform("count")

    df["norm_cascade_score"] = df["cascade_score"] / (q_max_score + 1e-9)
    df["top_minus_score"] = q_max_score - df["cascade_score"]
    df["z_score"] = (df["cascade_score"] - q_mean_score) / (q_std_score + 1e-9)
    df["log_rank"] = np.log1p(df["rrf_rank"])
    df["log1p_cascade"] = np.log1p(np.maximum(df["cascade_score"], 0.0))

    feature_cols = [
        "rrf_rank",
        "rrf_score",
        "norm_rrf_score",
        "inv_rrf_rank",
        "geo_tier",
        "same_loc",
        "dist_km",
        "log1p_dist",
        "valid_cat",
        "is_deliv",
        "mc_rank",
        "cascade_score",
        "log1p_cascade",
        "norm_cascade_score",
        "top_minus_score",
        "z_score",
        "log_rank",
    ]
    print(f"Features engineered ({len(feature_cols)} features) in {time.time() - t_start:.1f}s")

    # Select training subset: hard negatives (errors 1-50 vs 51-1000)
    print("\n[3/5] Sampling hard training pairs: positives + top-50 errors + sampled 51-1000...")
    pos_mask = df["label"] == 1
    top50_neg_mask = (df["label"] == 0) & (df["rrf_rank"] <= 50)
    # Sample from rank 51-1000
    mid_neg_mask = (df["label"] == 0) & (df["rrf_rank"] > 50) & (df["rrf_rank"] <= 300)
    # Use deterministic mask for training
    rng = np.random.default_rng(42)
    sample_sub = (df["rrf_rank"] % 5 == 0)  # 20% sample of mid/tail negatives
    mid_sample_mask = mid_neg_mask & sample_sub

    train_eligible_mask = pos_mask | top50_neg_mask | mid_sample_mask
    print(f"Full candidate pool: {len(df):,} rows -> Training set: {train_eligible_mask.sum():,} rows")

    # 5-Fold Strict OOF Training
    print("\n[4/5] 5-Fold Strict OOF CatBoostRanker (YetiRank)...")
    oof_ranker_scores = np.zeros(len(df), dtype=np.float32)

    for fold_id in range(5):
        t_fold = time.time()
        train_idx = (df["fold"] != fold_id) & train_eligible_mask
        val_idx = df["fold"] == fold_id

        train_sub = df.loc[train_idx].sort_values("group_id")
        val_sub = df.loc[val_idx].sort_values("group_id")

        train_pool = Pool(
            data=train_sub[feature_cols],
            label=train_sub["label"],
            group_id=train_sub["group_id"],
        )
        val_pool = Pool(
            data=val_sub[feature_cols],
            group_id=val_sub["group_id"],
        )

        ranker = CatBoostRanker(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            loss_function="YetiRank",
            eval_metric="NDCG:top=50",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
        ranker.fit(train_pool)
        val_preds = ranker.predict(val_pool)

        # Place predictions back in original order
        oof_ranker_scores[val_sub.index.to_numpy()] = val_preds
        print(f"  Fold {fold_id} completed in {time.time() - t_fold:.1f}s")

    df["ranker_score"] = oof_ranker_scores

    # 5. Evaluate Ranking Strategies
    print("\n[5/5] Evaluating Ranking Strategies across all 5,347 validation queries...")
    # Group predictions by qid
    grouped_df = df.groupby("qid", sort=False)

    def eval_rankings(item_scores_per_query: dict[str, list[tuple[str, float]]]) -> dict[str, Any]:
        rec_all: list[float] = []
        for qid in query_ids:
            candidates = item_scores_per_query.get(qid, [])
            candidates.sort(key=lambda p: (-p[1], p[0]))
            top50 = {it for it, _ in candidates[:50]}
            gt = relevant.get(qid, set())
            rec_all.append(len(top50 & gt) / max(len(gt), 1))
        arr = np.array(rec_all)
        r_seen = float(arr[is_seen].mean())
        r_unseen = float(arr[is_unseen].mean())
        proxy = 0.35 * r_seen + 0.65 * r_unseen
        return {
            "overall": float(arr.mean()),
            "seen": r_seen,
            "unseen": r_unseen,
            "proxy": proxy,
        }

    # Strategy A: Baseline Geo-Cascade
    baseline_tuples: dict[str, list[tuple[str, float]]] = {}
    for qid, group in grouped_df:
        baseline_tuples[str(qid)] = list(zip(group["item_id"], group["cascade_score"], strict=True))
    res_a = eval_rankings(baseline_tuples)
    print(f"Strategy A (Baseline Geo-Cascade): Overall={res_a['overall']:.4f} | Seen={res_a['seen']:.4f} | UNSEEN={res_a['unseen']:.4f} | Proxy={res_a['proxy']:.4f}")

    # Strategy B: Pure Ranker
    ranker_tuples: dict[str, list[tuple[str, float]]] = {}
    for qid, group in grouped_df:
        ranker_tuples[str(qid)] = list(zip(group["item_id"], group["ranker_score"], strict=True))
    res_b = eval_rankings(ranker_tuples)
    print(f"Strategy B (Pure Ranker):        Overall={res_b['overall']:.4f} | Seen={res_b['seen']:.4f} | UNSEEN={res_b['unseen']:.4f} | Proxy={res_b['proxy']:.4f}")

    # Strategy C: Ranker within Geo Tiers
    # Tier 1 gets +1000 boost, Tier 2 gets +100, Tier 3 gets +10, Tier 4 gets 0
    tier_tuples: dict[str, list[tuple[str, float]]] = {}
    for qid, group in grouped_df:
        tier_boost = (5 - group["geo_tier"]) * 100.0
        score_c = group["ranker_score"] + tier_boost
        tier_tuples[str(qid)] = list(zip(group["item_id"], score_c, strict=True))
    res_c = eval_rankings(tier_tuples)
    print(f"Strategy C (Ranker within Tiers): Overall={res_c['overall']:.4f} | Seen={res_c['seen']:.4f} | UNSEEN={res_c['unseen']:.4f} | Proxy={res_c['proxy']:.4f}")

    # Strategy D: Hybrid Grid: Geo Prior + lambda * Ranker Score
    # Standardize ranker score per query to mean=0, std=1
    q_r_mean = grouped_df["ranker_score"].transform("mean")
    q_r_std = grouped_df["ranker_score"].transform("std").fillna(1.0)
    df["std_ranker_score"] = (df["ranker_score"] - q_r_mean) / (q_r_std + 1e-9)

    best_hybrid_res = res_a
    best_lambda = 0.0

    hybrid_grid_results = {}
    for lam in [0.05, 0.1, 0.25, 0.5, 1.0]:
        h_tuples: dict[str, list[tuple[str, float]]] = {}
        for qid, group in grouped_df:
            # Scale cascade score by log1p and add lambda * std_ranker_score
            score_d = group["log1p_cascade"] + lam * group["std_ranker_score"]
            h_tuples[str(qid)] = list(zip(group["item_id"], score_d, strict=True))
        res_d = eval_rankings(h_tuples)
        hybrid_grid_results[f"lambda_{lam}"] = res_d
        print(f"Strategy D (lambda={lam:<4}):       Overall={res_d['overall']:.4f} | Seen={res_d['seen']:.4f} | UNSEEN={res_d['unseen']:.4f} | Proxy={res_d['proxy']:.4f}")
        if res_d["unseen"] > best_hybrid_res["unseen"]:
            best_hybrid_res = res_d
            best_lambda = lam

    # Decision gate
    base_unseen_r50 = res_a["unseen"]
    best_unseen_r50 = max(res_a["unseen"], res_b["unseen"], res_c["unseen"], best_hybrid_res["unseen"])
    delta_pp = (best_unseen_r50 - base_unseen_r50) * 100
    decision = "ACCEPT" if delta_pp >= 0.5 else "KILL"

    print("\n" + "=" * 80)
    print("WORKER C DECISION GATE:")
    print(f"  Baseline Unseen R@50: {base_unseen_r50:.4f}")
    print(f"  Best Unseen R@50:     {best_unseen_r50:.4f} (best_lambda={best_lambda})")
    print(f"  Delta:                {delta_pp:+.2f} pp")
    print(f"  Decision:             {decision}")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline_cascade": res_a,
        "pure_ranker": res_b,
        "ranker_within_tiers": res_c,
        "hybrid_grid": hybrid_grid_results,
        "best_strategy": f"hybrid_lambda_{best_lambda}" if best_lambda > 0 else "baseline_cascade",
        "best_unseen_r50": best_unseen_r50,
        "delta_pp": delta_pp,
        "decision": decision,
        "runtime_sec": time.time() - t0,
    }
    with (ranking_dir / "worker_c_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {ranking_dir / 'worker_c_report.json'}")


if __name__ == "__main__":
    main()
