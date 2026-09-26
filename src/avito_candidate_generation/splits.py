"""Leakage-safe deterministic item splits and ground truth artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def assign_item_splits(
    item_ids: list[str],
    *,
    seed: int = 42,
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    if len(proportions) != 3 or abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError("proportions must sum to 1")
    rows = []
    labels = ("train", "validation", "test")
    for item_id in sorted(set(item_ids)):
        digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        cumulative = 0.0
        split = labels[-1]
        for label, proportion in zip(labels, proportions):
            cumulative += proportion
            if value < cumulative:
                split = label
                break
        rows.append((item_id, split))
    return pd.DataFrame(rows, columns=["item_id", "split"])


def build_ground_truth(
    interactions: pd.DataFrame, item_splits: pd.DataFrame
) -> pd.DataFrame:
    required = {"internal_query_id", "item_id"}
    if not required.issubset(interactions.columns):
        raise ValueError("interactions require internal_query_id and item_id")
    if bool(item_splits["item_id"].duplicated().any()):
        raise ValueError("item splits contain duplicate item IDs")
    result = (
        interactions[["internal_query_id", "item_id"]]
        .drop_duplicates()
        .merge(item_splits, on="item_id", how="left", validate="many_to_one")
    )
    if bool(result["split"].isna().any()):
        raise ValueError("interaction references unknown split item")
    return pd.DataFrame(result[["split", "internal_query_id", "item_id"]])


def write_split_artifacts(
    interactions: pd.DataFrame,
    item_ids: list[str],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    item_splits = assign_item_splits(item_ids, seed=seed)
    ground_truth = build_ground_truth(interactions, item_splits)
    slices = ground_truth.groupby(["split", "internal_query_id"], as_index=False).agg(
        relevant_count=("item_id", "nunique")
    )
    slices = slices.assign(is_multi_positive=slices["relevant_count"] > 1)
    paths = {
        "item_splits": out / "item_splits.parquet",
        "ground_truth": out / "ground_truth.parquet",
        "query_slices": out / "query_slices.parquet",
    }
    item_splits.to_parquet(paths["item_splits"], index=False)
    ground_truth.to_parquet(paths["ground_truth"], index=False)
    slices.to_parquet(paths["query_slices"], index=False)
    report = out / "split_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": seed,
                "item_counts": item_splits["split"].value_counts().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    paths["report"] = report
    return paths
