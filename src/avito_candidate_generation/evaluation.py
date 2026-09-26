"""Canonical full-corpus query-mean recall evaluator."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .candidates import CandidateError, validate_candidates


def recall_at_k(predictions: pd.DataFrame, ground_truth: pd.DataFrame, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    required = {"internal_query_id", "item_id"}
    if not required.issubset(ground_truth.columns):
        raise CandidateError("ground truth requires internal_query_id and item_id")
    if ground_truth.duplicated(["internal_query_id", "item_id"]).any():
        raise CandidateError("duplicate ground-truth pair")
    if predictions.empty:
        return 0.0
    validate_candidates(predictions)
    query_ids = list(dict.fromkeys(ground_truth["internal_query_id"].tolist()))
    if not set(predictions["internal_query_id"]).issubset(set(query_ids)):
        raise CandidateError("prediction-only query")
    relevant = {
        q: set(g["item_id"]) for q, g in ground_truth.groupby("internal_query_id")
    }
    values: list[float] = []
    try:
        ranked = predictions.sort_values(["internal_query_id", "rank", "item_id"], kind="mergesort").drop_duplicates(["internal_query_id", "item_id"])  # pyright: ignore[reportCallIssue]
    except (KeyError, TypeError) as exc:
        raise CandidateError("invalid prediction ordering columns") from exc
    for query_id in query_ids:
        got_values = ranked.loc[ranked["internal_query_id"] == query_id, "item_id"].head(k).astype(str).tolist()
        got = set(got_values)
        try:
            values.append(float(len(got & relevant[query_id])) / float(len(relevant[query_id])))
        except (KeyError, ZeroDivisionError, TypeError) as exc:
            raise CandidateError("invalid ground-truth relevance") from exc
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_candidates(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ks: tuple[int, ...] = (10, 20, 50, 100, 200, 500),
    *,
    slices: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    metrics = {f"recall@{k}": recall_at_k(predictions, ground_truth, k) for k in ks}
    try:
        query_count = len(set(ground_truth["internal_query_id"].astype(str).tolist()))
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateError("invalid ground-truth query IDs") from exc
    return {
        "schema_version": 1,
        "metrics": metrics,
        "counts": {"queries": query_count, "prediction_rows": len(predictions)},
        "slices": slices or {},
        "status": "PASS",
    }
