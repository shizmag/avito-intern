"""Canonical full-corpus query-mean Recall@K evaluator and CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    validate_candidates(predictions)
    query_ids = list(dict.fromkeys(ground_truth["internal_query_id"].astype(str).tolist()))
    if not set(predictions["internal_query_id"].astype(str)).issubset(set(query_ids)):
        raise CandidateError("prediction-only query")
    relevant = {str(q): set(g["item_id"].astype(str)) for q, g in ground_truth.groupby("internal_query_id")}
    ranked = predictions.sort_values(["internal_query_id", "rank", "item_id"], kind="mergesort").drop_duplicates(["internal_query_id", "item_id"])  # pyright: ignore[reportCallIssue]
    values: list[float] = []
    for query_id in query_ids:
        got = set(ranked.loc[ranked["internal_query_id"].astype(str) == query_id, "item_id"].head(k).astype(str).tolist())
        values.append(len(got & relevant[query_id]) / len(relevant[query_id]))
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_candidates(predictions: pd.DataFrame, ground_truth: pd.DataFrame, ks: tuple[int, ...] = (10, 20, 50, 100, 200, 500), *, stage: str = "evaluation", split: str = "validation", candidate_fingerprint: str | None = None, ground_truth_fingerprint: str | None = None) -> dict[str, Any]:
    metrics = {f"recall@{k}": recall_at_k(predictions, ground_truth, k) for k in ks}
    return {"schema_version": 1, "stage": stage, "split": split, "metrics": metrics, "slices": {}, "counts": {"queries": len(set(ground_truth["internal_query_id"].astype(str))), "ground_truth_rows": len(ground_truth), "prediction_rows": len(predictions)}, "fingerprints": {"candidates": candidate_fingerprint, "ground_truth": ground_truth_fingerprint}, "timings": {}, "status": "PASS"}


def _fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.sort_values(list(frame.columns), kind="mergesort").to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("unable to fingerprint empty serialization")
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluate", choices=("evaluate",))
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ground_truth = pd.read_parquet(args.ground_truth)
    candidates = pd.read_parquet(args.candidates)
    if "split" in ground_truth.columns:
        ground_truth = pd.DataFrame(ground_truth.loc[ground_truth["split"] == args.split])
    result = evaluate_candidates(candidates, ground_truth, stage="evaluation", split=args.split, candidate_fingerprint=_fingerprint(candidates), ground_truth_fingerprint=_fingerprint(ground_truth))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
