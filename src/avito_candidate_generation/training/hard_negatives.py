"""Leakage-safe weak hard-negative mining and validation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def mine_hard_negatives(
    candidates: pd.DataFrame, known_positive: pd.DataFrame, *, max_per_query: int = 20
) -> pd.DataFrame:
    required = {"internal_query_id", "item_id", "source", "rank", "score"}
    if not required.issubset(candidates.columns):
        raise ValueError("candidate columns missing")
    if max_per_query < 1:
        raise ValueError("max_per_query must be positive")
    positive_pairs = set(
        zip(
            known_positive["internal_query_id"].astype(str),
            known_positive["item_id"].astype(str),
        )
    )
    work = candidates.copy()
    work["_pair"] = list(
        zip(work["internal_query_id"].astype(str), work["item_id"].astype(str))
    )
    work = work[~work["_pair"].isin(list(positive_pairs))].drop(columns="_pair")
    work = work.sort_values(["internal_query_id", "source", "rank", "item_id"], kind="mergesort")  # pyright: ignore[reportCallIssue]
    work = work.drop_duplicates(["internal_query_id", "item_id"], keep="first")
    work["iteration"] = 1
    work["sampling_order"] = work.groupby("internal_query_id").cumcount() + 1
    work["label"] = 0
    if "sources" not in work:
        work["sources"] = work["source"].map(lambda value: [str(value)])
    return (
        work.groupby("internal_query_id", group_keys=False)
        .head(max_per_query)
        .reset_index(drop=True)
    )


def validate_negatives(
    negatives: pd.DataFrame,
    known_positive: pd.DataFrame,
    *,
    max_per_query: int = 20,
    known_item_ids: set[str] | None = None,
) -> dict[str, int | bool]:
    if negatives.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate negative pair")
    positive_pairs = set(
        zip(
            known_positive["internal_query_id"].astype(str),
            known_positive["item_id"].astype(str),
        )
    )
    if any(
        pair in positive_pairs
        for pair in zip(
            negatives["internal_query_id"].astype(str), negatives["item_id"].astype(str)
        )
    ):
        raise ValueError("known positive collision")
    if known_item_ids is not None and not set(negatives["item_id"]).issubset(
        known_item_ids
    ):
        raise ValueError("unknown negative item")
    counts = negatives.groupby("internal_query_id").size()
    if bool((counts > max_per_query).any()):
        raise ValueError("per-query negative limit exceeded")
    if "source" not in negatives.columns and "sources" not in negatives.columns:
        raise ValueError("missing negative provenance")
    try:
        query_count = len(set(negatives["internal_query_id"].astype(str).tolist()))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid negative query IDs") from exc
    return {"rows": int(len(negatives)), "queries": query_count, "collisions": 0, "status": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mine", "validate"))
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--positives", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/hard_negatives/negatives.parquet"),
    )
    parser.add_argument("--max-per-query", type=int, default=20)
    args = parser.parse_args()
    if args.candidates is None or args.positives is None:
        raise SystemExit("--candidates and --positives required")
    candidates = pd.read_parquet(args.candidates)
    positives = pd.read_parquet(args.positives)
    if args.command == "mine":
        result = mine_hard_negatives(
            candidates, positives, max_per_query=args.max_per_query
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
    else:
        result = validate_negatives(
            pd.read_parquet(args.output), positives, max_per_query=args.max_per_query
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
