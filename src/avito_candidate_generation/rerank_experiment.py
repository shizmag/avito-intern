"""Orchestrator for representative and full Jina Reranker evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import load_config
from .evaluation import recall_at_k
from .fusion.catboost import load_selector
from .fusion.rrf import reciprocal_rank_fusion
from .reranker import (
    JinaReranker,
    rank_reranked_candidates,
    score_candidate_pool,
)
from .retrievers.dense import TransformerTextEncoder, load_embedding_artifact
from .retrievers.two_tower import TwoTowerModel
from .workflow import (
    _frame,
    _load_workflow_data,
    _retrieve_sources,
    prepare_selector_features,
    rank_with_selector,
)

logger = logging.getLogger(__name__)


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def sample_representative_queries(
    raw_queries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    sample_size: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Select deterministic stratified sample of validation queries."""
    if raw_queries.empty or "internal_query_id" not in raw_queries.columns:
        return raw_queries.copy()

    if sample_size >= len(raw_queries) or sample_size <= 0:
        return cast(
            pd.DataFrame,
            raw_queries.sort_values("internal_query_id", kind="mergesort").reset_index(
                drop=True
            ),  # pyright: ignore[reportCallIssue]
        )

    gt_pairs = set(
        zip(
            ground_truth["internal_query_id"].astype(str),
            ground_truth["item_id"].astype(str),
            strict=True,
        )
    )
    q_counts: dict[str, int] = {}
    for qid, _ in gt_pairs:
        q_counts[qid] = q_counts.get(qid, 0) + 1

    rows: list[dict[str, Any]] = []
    for _, row in raw_queries.iterrows():
        qid = str(row["internal_query_id"])
        n_pos = q_counts.get(qid, 0)
        if n_pos == 0:
            continue
        params_str = str(row.get("search_infm_params_text", "") or "").strip()
        has_filter = bool(params_str and params_str.lower() != "nan")
        query_str = str(row.get("search_query", "") or "").strip()
        q_len = len(query_str.split())
        len_cat = "short" if q_len <= 2 else ("medium" if q_len <= 4 else "long")
        stratum = f"{n_pos > 1}_{has_filter}_{len_cat}"
        rows.append({"internal_query_id": qid, "stratum": stratum})

    if not rows:
        return cast(pd.DataFrame, raw_queries.head(sample_size).copy())

    df_strata = pd.DataFrame(rows)
    rng = random.Random(seed)
    sampled_qids: list[str] = []

    groups = df_strata.groupby("stratum")
    frac = (sample_size / len(df_strata)) if len(df_strata) > 0 else 1.0
    for _, group in groups:
        g_qids = group["internal_query_id"].tolist()
        rng.shuffle(g_qids)
        k = max(1, _safe_int(round(len(g_qids) * frac), 1))
        sampled_qids.extend(g_qids[:k])

    rng.shuffle(sampled_qids)
    if len(sampled_qids) > sample_size:
        sampled_qids = sampled_qids[:sample_size]
    elif len(sampled_qids) < sample_size:
        remaining = [
            q
            for q in df_strata["internal_query_id"].tolist()
            if q not in set(sampled_qids)
        ]
        rng.shuffle(remaining)
        sampled_qids.extend(remaining[: sample_size - len(sampled_qids)])

    sampled_list = list(dict.fromkeys(sampled_qids))
    result = cast(
        pd.DataFrame,
        raw_queries[
            raw_queries["internal_query_id"].astype(str).isin(sampled_list)
        ].copy(),
    )
    return cast(
        pd.DataFrame,
        result.sort_values("internal_query_id", kind="mergesort").reset_index(
            drop=True
        ),  # pyright: ignore[reportCallIssue]
    )


def compute_metrics_suite(
    ranked_candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ks: Sequence[int] = (10, 20, 50, 100, 200, 500),
) -> dict[str, float]:
    """Compute Recall@K across multiple K values."""
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranked_candidates, ground_truth, k)
    return metrics


def run_reranker_validation(
    config_path: str | Path = "configs/selected.toml",
    *,
    sample_size: int = 500,
    candidate_k: int = 500,
    device: str = "auto",
    batch_size: int = 64,
    cache_dir: str | Path = "artifacts/reranker/cache",
    output_json: str | Path = "artifacts/metrics/reranker_validation.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Run end-to-end apples-to-apples validation comparing retrievers, RRF, CatBoost, and Jina."""
    config = load_config(config_path)
    data = _load_workflow_data(config)

    raw_queries = data.validation_queries
    raw_items = data.validation_items
    ground_truth = data.validation_ground_truth

    # 1. Sample queries
    logger.info("Sampling %d representative validation queries...", sample_size)
    sampled_raw_queries = sample_representative_queries(
        raw_queries, ground_truth, sample_size=sample_size, seed=seed
    )
    sampled_qids_list = list(
        dict.fromkeys(sampled_raw_queries["internal_query_id"].astype(str).tolist())
    )
    sampled_gt = _frame(
        ground_truth[
            ground_truth["internal_query_id"].astype(str).isin(sampled_qids_list)
        ].copy()
    )

    sampled_norm_queries = _frame(
        data.validation_queries[
            data.validation_queries["internal_query_id"]
            .astype(str)
            .isin(sampled_qids_list)
        ].copy()
    )

    # 2. Retrieve candidates from BM25, Dense E5, Two-Tower v1
    dense_cfg = config.values.get("dense", {})
    model_name = (
        str(dense_cfg.get("model_path", "artifacts/models/multilingual-e5-base"))
        if isinstance(dense_cfg, dict)
        else "artifacts/models/multilingual-e5-base"
    )
    max_length = _safe_int(
        dense_cfg.get("max_length", 128) if isinstance(dense_cfg, dict) else 128, 128
    )
    query_prefix = (
        str(dense_cfg.get("query_prefix", "query: "))
        if isinstance(dense_cfg, dict)
        else "query: "
    )

    encoder = TransformerTextEncoder(
        model_name,
        max_length=max_length,
        normalize=True,
        device=device,
    )
    e5_items = load_embedding_artifact(
        "artifacts/real_validation/embeddings/e5_base/validation"
    )
    if e5_items is None:
        raise FileNotFoundError("Missing precomputed E5 item embeddings")

    tower = TwoTowerModel.load("artifacts/real_validation/two_tower/v1/checkpoint.json")
    tt_items = load_embedding_artifact(
        "artifacts/real_validation/embeddings/two_tower/v1/validation"
    )
    if tt_items is None:
        raise FileNotFoundError("Missing precomputed Two-Tower item embeddings")

    logger.info(
        "Retrieving top-%d candidates for %d queries...",
        candidate_k,
        len(sampled_norm_queries),
    )
    t0_retrieval = time.perf_counter()
    category_policy = "none"
    selection_cfg = config.values.get("selection")
    if isinstance(selection_cfg, dict):
        category_policy = str(selection_cfg.get("category_policy", "none"))

    sources = _retrieve_sources(
        sampled_norm_queries,
        data.validation_items,
        encoder,
        e5_items,
        tower,
        tt_items,
        k=candidate_k,
        category_policy=category_policy,
        dense_query_prefix=query_prefix,
    )
    retrieval_sec = time.perf_counter() - t0_retrieval
    logger.info("Candidate retrieval finished in %.2fs", retrieval_sec)

    # Calculate baseline metrics
    bm25_cand = sources[0]
    dense_cand = sources[1]
    tt_cand = sources[2]

    bm25_metrics = compute_metrics_suite(bm25_cand, sampled_gt)
    dense_metrics = compute_metrics_suite(dense_cand, sampled_gt)
    tt_metrics = compute_metrics_suite(tt_cand, sampled_gt)

    all_candidates = _frame(pd.concat(sources, ignore_index=True))

    # Candidate pool coverage
    union_pairs = set(
        zip(
            all_candidates["internal_query_id"].astype(str),
            all_candidates["item_id"].astype(str),
            strict=True,
        )
    )
    gt_pairs = set(
        zip(
            sampled_gt["internal_query_id"].astype(str),
            sampled_gt["item_id"].astype(str),
            strict=True,
        )
    )
    full_union_coverage = (
        len(union_pairs & gt_pairs) / len(gt_pairs) if gt_pairs else 0.0
    )

    # RRF top-500 candidate pool
    fusion_cfg = config.values.get("fusion")
    rrf_k = _safe_int(
        fusion_cfg.get("rrf_k", 60) if isinstance(fusion_cfg, dict) else 60, 60
    )
    rrf_pool = reciprocal_rank_fusion(all_candidates, rrf_k=rrf_k, limit=candidate_k)
    rrf_metrics = compute_metrics_suite(rrf_pool, sampled_gt)

    rrf_pairs = set(
        zip(
            rrf_pool["internal_query_id"].astype(str),
            rrf_pool["item_id"].astype(str),
            strict=True,
        )
    )
    candidate_pool_coverage = (
        len(rrf_pairs & gt_pairs) / len(gt_pairs) if gt_pairs else 0.0
    )

    # CatBoost baseline on identical candidates
    catboost_metrics: dict[str, float] = {}
    cb_path = Path("artifacts/real_validation/catboost_selector")
    if not cb_path.is_file():
        cb_path = Path("artifacts/real_validation/catboost_selector.json")
    if cb_path.is_file():
        try:
            selector = load_selector(cb_path)
            raw_feat_cols = selector.get("feature_columns")
            feat_cols: list[str] = (
                [str(x) for x in raw_feat_cols]
                if isinstance(raw_feat_cols, list)
                else []
            )
            use_short_names = "e5_rank" in feat_cols
            source_names = (
                ("bm25", "e5", "tt")
                if use_short_names
                else ("bm25", "dense", "two_tower")
            )
            mapped_candidates = all_candidates.copy()
            if use_short_names:
                mapped_candidates["source"] = (
                    mapped_candidates["source"]
                    .astype(str)
                    .map(lambda s: {"dense": "e5", "two_tower": "tt"}.get(s, s))
                )
            catboost_ranked = rank_with_selector(
                mapped_candidates,
                selector,
                limit=50,
                missing_rank=candidate_k + 1,
                source_names=source_names,
            )
            catboost_metrics = compute_metrics_suite(
                catboost_ranked, sampled_gt, ks=[10, 20, 50]
            )
        except Exception as exc:
            logger.warning("CatBoost baseline evaluation skipped: %s", exc)

    # 3. Jina Reranker Provisioning & Inference
    reranker_cfg = config.values.get("reranker")
    reranker_path = (
        str(
            reranker_cfg.get(
                "local_path",
                "/Volumes/happy-disk/models/reranker/jina-reranker-v2-base-multilingual",
            )
        )
        if isinstance(reranker_cfg, dict)
        else "/Volumes/happy-disk/models/reranker/jina-reranker-v2-base-multilingual"
    )
    r_max_len = _safe_int(
        reranker_cfg.get("max_length", 256) if isinstance(reranker_cfg, dict) else 256,
        256,
    )
    reranker_model = JinaReranker(
        reranker_path,
        max_length=r_max_len,
        device=device,
    )

    logger.info("Scoring candidate pool with Jina reranker...")
    t0_rerank = time.perf_counter()

    def on_progress(q_done: int, q_tot: int, p_done: int, p_tot: int) -> None:
        pct = (100.0 * q_done / q_tot) if q_tot > 0 else 100.0
        print(
            f"Reranking progress: queries {q_done}/{q_tot} ({pct:.1f}%), pairs {p_done}/{p_tot}",
            flush=True,
        )

    scored_pool = score_candidate_pool(
        rrf_pool,
        sampled_raw_queries,
        raw_items,
        reranker_model,
        pair_batch_size=batch_size,
        query_chunk_size=50,
        cache_dir=cache_dir,
        progress_callback=on_progress,
    )
    rerank_sec = time.perf_counter() - t0_rerank
    pairs_per_sec = (len(scored_pool) / rerank_sec) if rerank_sec > 0 else 0.0
    logger.info(
        "Scored %d pairs in %.2fs (%.1f pairs/sec)",
        len(scored_pool),
        rerank_sec,
        pairs_per_sec,
    )

    # Jina standalone ranking
    jina_ranked = rank_reranked_candidates(
        scored_pool, score_column="jina_score", limit=50, source_name="jina_reranker"
    )
    jina_metrics = compute_metrics_suite(jina_ranked, sampled_gt)

    # 4. Candidate Pool Depth Ablation (top 100, 200, 300, 500)
    depth_ablation: dict[str, dict[str, float]] = {}
    for depth in [100, 200, 300, 500]:
        sub_pool = rrf_pool.groupby("internal_query_id", group_keys=False).head(depth)
        sub_pairs = set(
            zip(
                sub_pool["internal_query_id"].astype(str),
                sub_pool["item_id"].astype(str),
                strict=True,
            )
        )
        sub_cov = len(sub_pairs & gt_pairs) / len(gt_pairs) if gt_pairs else 0.0

        sub_item_list = list(dict.fromkeys(sub_pool["item_id"].astype(str).tolist()))
        sub_qid_list = list(
            dict.fromkeys(sub_pool["internal_query_id"].astype(str).tolist())
        )
        sub_scored = _frame(
            scored_pool[
                scored_pool["item_id"].astype(str).isin(sub_item_list)
                & scored_pool["internal_query_id"].astype(str).isin(sub_qid_list)
            ].copy()
        )
        sub_jina_ranked = rank_reranked_candidates(sub_scored, limit=50)
        sub_r50 = recall_at_k(sub_jina_ranked, sampled_gt, 50)
        depth_ablation[f"top_{depth}"] = {
            "candidate_pool_recall": round(sub_cov, 6),
            "reranked_recall@50": round(sub_r50, 6),
            "pairs": round(len(sub_pool) * 1.0, 1),
        }

    # 5. Hybrid Jina + CatBoost feature experiment
    hybrid_metrics: dict[str, float] = {}
    try:
        from catboost import CatBoostClassifier  # type: ignore[import-not-found]

        selector_feats = prepare_selector_features(
            all_candidates, missing_rank=candidate_k + 1
        )
        jina_scores_map = cast(
            pd.DataFrame,
            scored_pool[["internal_query_id", "item_id", "jina_score"]].drop_duplicates(
                subset=["internal_query_id", "item_id"]  # pyright: ignore[reportCallIssue]
            ),
        )
        hybrid_feats = selector_feats.merge(
            jina_scores_map, on=["internal_query_id", "item_id"], how="left"
        )
        hybrid_feats["jina_score"] = hybrid_feats["jina_score"].fillna(-999.0)

        gt_pairs_set = set(
            zip(
                sampled_gt["internal_query_id"].astype(str),
                sampled_gt["item_id"].astype(str),
                strict=True,
            )
        )
        hybrid_feats["label"] = [
            int((str(q), str(i)) in gt_pairs_set)
            for q, i in zip(
                hybrid_feats["internal_query_id"], hybrid_feats["item_id"], strict=True
            )
        ]

        tuning_qids: list[str] = []
        for q in sampled_qids_list:
            b = (
                int.from_bytes(
                    hashlib.sha256(f"hybrid:{q}".encode()).digest()[:8], "big"
                )
                % 5
            )
            if b == 0:
                tuning_qids.append(q)

        train_mask = hybrid_feats["internal_query_id"].astype(str).isin(tuning_qids)
        test_mask = ~train_mask

        feature_cols = [
            str(c)
            for c in hybrid_feats.columns
            if c not in {"internal_query_id", "item_id", "label"}
            and pd.api.types.is_numeric_dtype(hybrid_feats[c])
        ]

        train_data = hybrid_feats.loc[train_mask]
        test_data = _frame(hybrid_feats.loc[test_mask].copy())

        if (
            len(train_data) > 0
            and len(test_data) > 0
            and train_data["label"].nunique() > 1
        ):
            cb_hybrid = CatBoostClassifier(
                iterations=250,
                depth=7,
                learning_rate=0.08,
                loss_function="Logloss",
                random_seed=seed,
                verbose=False,
                thread_count=1,
            )
            cb_hybrid.fit(train_data[feature_cols], train_data["label"])
            probs = cb_hybrid.predict_proba(test_data[feature_cols])
            test_data["cb_pred"] = probs[:, 1]
            test_ranked = _frame(
                test_data.sort_values(
                    ["internal_query_id", "cb_pred", "item_id"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                .groupby("internal_query_id", group_keys=False)
                .head(50)  # pyright: ignore[reportCallIssue]
            )
            test_ranked["rank"] = (
                test_ranked.groupby("internal_query_id").cumcount() + 1
            )
            test_ranked["source"] = "catboost_jina_hybrid"
            test_ranked["score"] = test_ranked["cb_pred"]

            test_qids_list = list(
                dict.fromkeys(test_data["internal_query_id"].astype(str).tolist())
            )
            heldout_gt = _frame(
                sampled_gt[
                    sampled_gt["internal_query_id"].astype(str).isin(test_qids_list)
                ].copy()
            )
            hybrid_metrics = compute_metrics_suite(
                test_ranked, heldout_gt, ks=[10, 20, 50]
            )
    except Exception as exc:
        logger.warning("Hybrid CatBoost+Jina training skipped: %s", exc)

    # 6. Error Analysis: 4 quadrants + positive recovery position distribution
    rrf_top50 = rrf_pool.groupby("internal_query_id", group_keys=False).head(50)
    rrf_top50_pairs = set(
        zip(
            rrf_top50["internal_query_id"].astype(str),
            rrf_top50["item_id"].astype(str),
            strict=True,
        )
    )

    jina_top50 = jina_ranked.groupby("internal_query_id", group_keys=False).head(50)
    jina_top50_pairs = set(
        zip(
            jina_top50["internal_query_id"].astype(str),
            jina_top50["item_id"].astype(str),
            strict=True,
        )
    )

    rrf_hit_qids: set[str] = set()
    jina_hit_qids: set[str] = set()

    for _, row in sampled_gt.iterrows():
        pair = (str(row["internal_query_id"]), str(row["item_id"]))
        if pair in rrf_top50_pairs:
            rrf_hit_qids.add(pair[0])
        if pair in jina_top50_pairs:
            jina_hit_qids.add(pair[0])

    all_eval_qids = set(sampled_gt["internal_query_id"].astype(str).tolist())
    both_hit = len(rrf_hit_qids & jina_hit_qids)
    rrf_only_hit = len(rrf_hit_qids - jina_hit_qids)
    jina_only_hit = len(jina_hit_qids - rrf_hit_qids)
    both_miss = len(all_eval_qids - (rrf_hit_qids | jina_hit_qids))

    error_analysis = {
        "total_evaluated_queries": len(all_eval_qids),
        "both_hit": both_hit,
        "rrf_hit_jina_miss": rrf_only_hit,
        "rrf_miss_jina_hit_novel_recoveries": jina_only_hit,
        "both_miss": both_miss,
    }

    # Position analysis
    rrf_rank_map: dict[tuple[str, str], int] = {}
    for q, i, r in zip(
        rrf_pool["internal_query_id"],
        rrf_pool["item_id"],
        rrf_pool["rank"],
        strict=True,
    ):
        rrf_rank_map[(str(q), str(i))] = _safe_int(r, 999)

    position_buckets = {
        "1-50": 0,
        "51-100": 0,
        "101-200": 0,
        "201-300": 0,
        "301-500": 0,
        "outside_pool": 0,
    }
    for _, row in sampled_gt.iterrows():
        pair = (str(row["internal_query_id"]), str(row["item_id"]))
        if pair in jina_top50_pairs:
            orig_rank = rrf_rank_map.get(pair)
            if orig_rank is None:
                position_buckets["outside_pool"] += 1
            elif orig_rank <= 50:
                position_buckets["1-50"] += 1
            elif orig_rank <= 100:
                position_buckets["51-100"] += 1
            elif orig_rank <= 200:
                position_buckets["101-200"] += 1
            elif orig_rank <= 300:
                position_buckets["201-300"] += 1
            else:
                position_buckets["301-500"] += 1

    # 7. Summary & Result Payload
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "protocol": f"representative validation ({sample_size} queries)",
        "sample_size": sample_size,
        "n_gt_pairs": len(sampled_gt),
        "candidate_k": candidate_k,
        "hardware": {
            "device": str(reranker_model.device),
            "pair_batch_size": batch_size,
            "pairs_scored": len(scored_pool),
            "rerank_runtime_sec": round(rerank_sec, 2),
            "pairs_per_sec": round(pairs_per_sec, 1),
        },
        "upper_bounds": {
            "full_generator_union_coverage": round(full_union_coverage, 6),
            "candidate_pool_500_coverage": round(candidate_pool_coverage, 6),
        },
        "metrics": {
            "bm25": {k: round(v, 6) for k, v in bm25_metrics.items()},
            "dense_e5": {k: round(v, 6) for k, v in dense_metrics.items()},
            "two_tower_v1": {k: round(v, 6) for k, v in tt_metrics.items()},
            "rrf_60": {k: round(v, 6) for k, v in rrf_metrics.items()},
            "catboost": {k: round(v, 6) for k, v in catboost_metrics.items()},
            "jina_reranker": {k: round(v, 6) for k, v in jina_metrics.items()},
            "jina_catboost_hybrid": {k: round(v, 6) for k, v in hybrid_metrics.items()},
        },
        "depth_ablation": depth_ablation,
        "error_analysis": error_analysis,
        "recovered_positives_original_rrf_rank": position_buckets,
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved validation report to %s", out_file)
    return report
