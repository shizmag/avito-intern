"""Leakage-safe weak hard-negative mining and validation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
            strict=True,
        )
    )
    work = candidates.copy()
    work["_pair"] = list(
        zip(
            work["internal_query_id"].astype(str),
            work["item_id"].astype(str),
            strict=True,
        )
    )
    work = work[~work["_pair"].isin(list(positive_pairs))].drop(columns="_pair")
    work = work.sort_values(
        ["internal_query_id", "source", "rank", "item_id"], kind="mergesort"
    )  # pyright: ignore[reportCallIssue]
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
) -> dict[str, Any]:
    if negatives.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate negative pair")
    positive_pairs = set(
        zip(
            known_positive["internal_query_id"].astype(str),
            known_positive["item_id"].astype(str),
            strict=True,
        )
    )
    try:
        has_overlap = any(
            pair in positive_pairs
            for pair in zip(
                negatives["internal_query_id"].astype(str),
                negatives["item_id"].astype(str),
                strict=True,
            )
        )
    except Exception as exc:
        raise ValueError("unable to validate negative pairs") from exc
    if has_overlap:
        raise ValueError("hard negative contains known positive pair")
    per_query = negatives.groupby("internal_query_id").size()
    if (per_query > max_per_query).any():
        raise ValueError(f"exceeded max_per_query={max_per_query}")
    if known_item_ids is not None and not set(
        negatives["item_id"].astype(str)
    ).issubset(known_item_ids):
        raise ValueError("hard negative references unknown item_id")
    try:
        n_unique_q = int(negatives["internal_query_id"].nunique())  # pyright: ignore[reportArgumentType]
        n_total = len(negatives)
        max_q = int(per_query.max()) if not per_query.empty else 0  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError) as exc:
        raise ValueError("unable to compute negative statistics") from exc
    return {
        "status": "PASS",
        "unique_queries": n_unique_q,
        "total_negatives": n_total,
        "max_per_query": max_q,
        "valid": True,
    }


def write_negative_report(report: dict[str, int | bool], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, **report}
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mine", "validate"))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-query", type=int, default=20)
    args = parser.parse_args()
    candidates = pd.read_parquet(args.candidates)
    positive = pd.read_parquet(args.positive)
    if args.command == "mine":
        result = mine_hard_negatives(
            candidates, positive, max_per_query=args.max_per_query
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
    else:
        report = validate_negatives(
            candidates, positive, max_per_query=args.max_per_query
        )
        write_negative_report(report, args.output)


if __name__ == "__main__":
    main()
