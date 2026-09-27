"""Memory-bounded real-data validation helpers."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Sequence

import numpy as np
from pathlib import Path
from typing import Any

import pandas as pd
from .retrievers.exact_search import exact_top_k

from .config import load_config
from .retrievers.bm25 import BM25Index
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
    index = BM25Index.fit(items)
    metrics = stream_recall_at_k(
        queries,
        ground_truth,
        lambda batch, k: index.retrieve(batch, k=k),
        batch_size=batch_size,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method": "bm25",
        "split": "cold-item-validation",
        "category_policy": config.values.get("selection", {}).get(
            "category_policy", "unspecified"
        ),
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
