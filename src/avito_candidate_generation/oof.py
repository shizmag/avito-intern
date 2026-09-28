"""Leakage-audited OOF feature table assembly and verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_oof_table(
    features: pd.DataFrame, ground_truth: pd.DataFrame, *, fold: int
) -> pd.DataFrame:
    if features.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate feature pair")
    labels = (
        ground_truth[["internal_query_id", "item_id"]].drop_duplicates().assign(label=1)
    )
    result = features.merge(
        labels, on=["internal_query_id", "item_id"], how="left", validate="one_to_one"
    )
    result["label"] = result["label"].fillna(0).astype("int8")
    result["fold"] = fold
    if result.duplicated(["fold", "internal_query_id", "item_id"]).any():
        raise ValueError("duplicate OOF row")
    return result.sort_values(
        ["internal_query_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)


def verify_oof(table: pd.DataFrame, *, folds: int = 3) -> dict[str, int | bool]:
    required = {"fold", "internal_query_id", "item_id", "label"}
    if not required.issubset(table.columns):
        raise ValueError("OOF columns missing")
    if table.duplicated(["fold", "internal_query_id", "item_id"]).any():
        raise ValueError("duplicate OOF key")
    actual = set(table["fold"].astype(int).tolist())
    if actual != set(range(folds)):
        raise ValueError("OOF folds incomplete")
    if not set(table["label"].unique()).issubset({0, 1}):
        raise ValueError("invalid OOF labels")
    try:
        positives = int((table["label"] == 1).sum())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid OOF labels") from exc
    return {
        "rows": len(table),
        "folds": len(actual),
        "positives": positives,
        "status": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oof/oof_candidates_features.parquet"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    if args.command == "build":
        result = build_oof_table(
            pd.read_parquet(args.features),
            pd.read_parquet(args.ground_truth),
            fold=args.fold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
    else:
        print(json.dumps(verify_oof(pd.read_parquet(args.output), folds=args.folds)))


if __name__ == "__main__":
    main()
