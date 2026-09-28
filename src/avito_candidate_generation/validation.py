"""Memory-bounded real-data validation helpers."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .retrievers.bm25 import BM25Index
from .retrievers.exact_search import exact_top_k
from .retrievers.tfidf import TFIDFIndex
from .workflow import _compose_item_text, _compose_query_text, _prepare_workflow_paths


def stream_exact_recall_at_k(
    queries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    encode: Callable[[Sequence[str]], np.ndarray],
    item_embeddings: np.ndarray,
    item_ids: Sequence[str],
    *,
    ks: Sequence[int] = (10, 20, 50, 100, 200, 500),
    query_batch_size: int = 32,
    item_batch_size: int = 2048,
    source: str = "dense",
) -> dict[str, float]:
    """Evaluate exact dense retrieval without retaining all query candidates."""
    if not ks or min(ks) < 1 or query_batch_size < 1 or item_batch_size < 1:
        raise ValueError("ks and batch sizes must be positive")
    if ground_truth.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate ground-truth pair")
    relevant = {
        str(query_id): set(group["item_id"].astype(str))
        for query_id, group in ground_truth.groupby("internal_query_id")
    }
    query_ids = queries["internal_query_id"].astype(str).tolist()
    if set(relevant).difference(query_ids):
        raise ValueError("ground truth references query absent from validation queries")
    sums = dict.fromkeys(ks, 0.0)
    max_k = max(ks)
    for start in range(0, len(queries), query_batch_size):
        batch = queries.iloc[start : start + query_batch_size]
        vectors = np.asarray(
            encode(batch["text"].astype(str).tolist()), dtype=np.float32
        )
        candidates = exact_top_k(
            vectors,
            item_embeddings,
            list(item_ids),
            k=max_k,
            query_ids=batch["internal_query_id"].astype(str).tolist(),
            source=source,
            batch_size=query_batch_size,
            item_batch_size=item_batch_size,
        )
        for query_id, group in candidates.groupby("internal_query_id"):
            items = (
                group.sort_values(["rank", "item_id"], kind="mergesort")["item_id"]
                .astype(str)
                .tolist()
            )
            expected = relevant[str(query_id)]
            for k in ks:
                sums[k] += len(set(items[:k]) & expected) / len(expected)
    denominator = len(relevant)
    return {f"recall@{k}": sums[k] / denominator for k in ks}


def stream_recall_at_k(
    queries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    retrieve: Callable[[pd.DataFrame, int], pd.DataFrame],
    *,
    ks: Sequence[int] = (10, 20, 50, 100, 200, 500),
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate query-mean Recall@K without materializing all candidates."""
    if not ks or min(ks) < 1 or batch_size < 1:
        raise ValueError("ks must be positive and batch_size must be positive")
    if ground_truth.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate ground-truth pair")
    query_ids = ground_truth["internal_query_id"].astype(str).drop_duplicates().tolist()
    relevant = {
        str(query_id): set(group["item_id"].astype(str))
        for query_id, group in ground_truth.groupby("internal_query_id")
    }
    if not set(query_ids).issubset(set(queries["internal_query_id"].astype(str))):
        raise ValueError("ground truth references query absent from validation queries")
    sums = dict.fromkeys(ks, 0.0)
    max_k = max(ks)
    for start in range(0, len(queries), batch_size):
        batch = queries.iloc[start : start + batch_size]
        candidates = retrieve(batch, max_k)
        for query_id, group in candidates.groupby("internal_query_id"):
            query_key = str(query_id)
            if query_key not in relevant:
                raise ValueError("retrieval returned query outside ground truth")
            ranked = group.sort_values(["rank", "item_id"], kind="mergesort")
            items = ranked["item_id"].astype(str).drop_duplicates().tolist()
            expected = relevant[query_key]
            for k in ks:
                sums[k] += len(set(items[:k]) & expected) / len(expected)
    denominator = len(query_ids)
    return {f"recall@{k}": sums[k] / denominator for k in ks}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_metric_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        {
            "method": payload["method"],
            "split": payload["split"],
            "metric": metric,
            "value": value,
            "n_queries": payload["n_queries"],
            "n_items": payload["n_items"],
            "candidate_k": payload["candidate_k"],
            "status": payload["status"],
        }
        for metric, value in payload["metrics"].items()
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_bm25_validation(
    config_path: str | Path = "configs/selected.toml",
    *,
    output: str | Path = "artifacts/metrics/real_validation.json",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run honest cold-item BM25 validation with bounded candidate memory."""
    config = load_config(config_path)
    paths = _prepare_workflow_paths(config, config.values["data"])
    queries = _compose_query_text(pd.read_parquet(paths["validation_queries"]))
    items = _compose_item_text(pd.read_parquet(paths["validation_items"]))
    ground_truth = pd.read_parquet(paths["validation_ground_truth"])
    category_policy = str(
        config.values.get("selection", {}).get("category_policy", "none")
    )
    if category_policy not in {"none", "hard", "fallback"}:
        raise ValueError("category_policy must be one of none, hard, fallback")
    global_index = BM25Index.fit(items)
    category_indexes = {
        str(category): BM25Index.fit(group)
        for category, group in items.dropna(subset=["item_category_id"]).groupby(
            "item_category_id", sort=False
        )
    }

    def retrieve_batch(batch: pd.DataFrame, k: int) -> pd.DataFrame:
        if category_policy == "none":
            return global_index.retrieve(batch, k=k)
        candidates: list[pd.DataFrame] = []
        for category, group in batch.groupby(
            "search_category", sort=False, dropna=False
        ):
            category_key = (
                None if category is None or str(category) == "nan" else str(category)
            )
            index = (
                category_indexes.get(category_key) if category_key is not None else None
            )
            if index is None:
                if category_policy == "hard":
                    continue
                index = global_index
            result = index.retrieve(group, k=k)
            if result.empty and category_policy == "fallback":
                result = global_index.retrieve(group, k=k)
            candidates.append(result)
        if not candidates:
            return pd.DataFrame(
                columns=[
                    "internal_query_id",
                    "item_id",
                    "source",
                    "score",
                    "rank",
                ]
            )
        return pd.concat(candidates, ignore_index=True)

    metrics = stream_recall_at_k(
        queries,
        ground_truth,
        retrieve_batch,
        batch_size=batch_size,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method": "bm25",
        "split": "cold-item-validation",
        "category_policy": category_policy,
        "n_queries": ground_truth["internal_query_id"].nunique(),
        "n_items": len(items),
        "n_relevant": len(ground_truth),
        "candidate_k": 500,
        "batch_size": batch_size,
        "metrics": metrics,
        "config_hash": config.hash,
        "reason": "BM25-only real validation; neural ablations not run by this command",
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(target, payload)
    _write_metric_csv(target.with_suffix(".csv"), payload)
    return payload


def run_tfidf_validation(
    config_path: str | Path = "configs/selected.toml",
    *,
    output: str | Path = "artifacts/metrics/tfidf_validation.json",
    subset_queries: int | None = 1000,
    ks: Sequence[int] = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    batch_size: int = 256,
    sublinear_tf: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run TF-IDF validation with configurable query subset and Recall@K steps."""
    config = load_config(config_path)
    paths = _prepare_workflow_paths(config, config.values["data"])
    raw_queries = pd.read_parquet(paths["validation_queries"])
    ground_truth = pd.read_parquet(paths["validation_ground_truth"])
    items = _compose_item_text(pd.read_parquet(paths["validation_items"]))

    if subset_queries is not None and 0 < subset_queries < len(raw_queries):
        from .rerank_experiment import sample_representative_queries

        queries_sample = sample_representative_queries(
            raw_queries, ground_truth, sample_size=subset_queries, seed=seed
        )
        queries = _compose_query_text(queries_sample)
        valid_qids = set(queries["internal_query_id"])
        ground_truth = ground_truth[
            ground_truth["internal_query_id"].isin(valid_qids)
        ].copy()
    else:
        queries = _compose_query_text(raw_queries)

    category_policy = str(
        config.values.get("selection", {}).get("category_policy", "none")
    )
    if category_policy not in {"none", "hard", "fallback"}:
        raise ValueError("category_policy must be one of none, hard, fallback")

    global_index = TFIDFIndex.fit(items, sublinear_tf=sublinear_tf)
    category_indexes = (
        {
            str(category): TFIDFIndex.fit(group, sublinear_tf=sublinear_tf)
            for category, group in items.dropna(subset=["item_category_id"]).groupby(
                "item_category_id", sort=False
            )
        }
        if category_policy != "none"
        else {}
    )

    def retrieve_batch(batch: pd.DataFrame, k: int) -> pd.DataFrame:
        if category_policy == "none":
            return global_index.retrieve(batch, k=k)
        candidates: list[pd.DataFrame] = []
        for category, group in batch.groupby(
            "search_category", sort=False, dropna=False
        ):
            category_key = (
                None if category is None or str(category) == "nan" else str(category)
            )
            index = (
                category_indexes.get(category_key) if category_key is not None else None
            )
            if index is None:
                if category_policy == "hard":
                    continue
                index = global_index
            result = index.retrieve(group, k=k)
            if result.empty and category_policy == "fallback":
                result = global_index.retrieve(group, k=k)
            candidates.append(result)
        if not candidates:
            return pd.DataFrame(
                columns=[
                    "internal_query_id",
                    "item_id",
                    "source",
                    "score",
                    "rank",
                ]
            )
        return pd.concat(candidates, ignore_index=True)

    metrics = stream_recall_at_k(
        queries,
        ground_truth,
        retrieve_batch,
        ks=ks,
        batch_size=batch_size,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method": "tfidf",
        "split": (
            "cold-item-validation-subset" if subset_queries else "cold-item-validation"
        ),
        "category_policy": category_policy,
        "sublinear_tf": sublinear_tf,
        "n_queries": ground_truth["internal_query_id"].nunique(),
        "n_items": len(items),
        "n_relevant": len(ground_truth),
        "candidate_k": max(ks),
        "batch_size": batch_size,
        "metrics": metrics,
        "config_hash": config.hash,
        "reason": "TF-IDF baseline validation with step 50 recall@k",
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(target, payload)
    _write_metric_csv(target.with_suffix(".csv"), payload)
    return payload
