"""Prepare strict unseen validation split and cached artifacts for all workers."""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.research import (
    CONTEXT_COLUMNS,
    context_folds,
    context_frame,
    ground_truth_map,
)


def normalize_query(s: Any) -> str:
    """Strict query normalization: NFKC, lowercase, ё -> е, whitespace normalization."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    return " ".join(s.split())


def main() -> None:
    t0 = time.time()
    out_dir = Path("artifacts/unseen_push/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 0: BUILD STRICT UNSEEN VALIDATION SPLIT AND CACHED DATASETS")
    print("=" * 80)

    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    # Split into train (folds 1..4) and val (fold 0)
    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0].copy()
    val_with_bm_pos = val_rows[
        val_rows["item_id"].astype(str).isin(benchmark_ids_set)
    ].copy()

    # Normalize queries
    train_part["search_query_norm"] = train_part["search_query"].apply(normalize_query)
    val_with_bm_pos["search_query_norm"] = val_with_bm_pos["search_query"].apply(
        normalize_query
    )

    train_norm_set = set(train_part["search_query_norm"].unique())
    # Exclude empty string from matching if any
    train_norm_set.discard("")

    context_cols = [*CONTEXT_COLUMNS, "search_query_norm", "internal_query_id"]
    val_contexts = (
        val_with_bm_pos[context_cols].drop_duplicates().reset_index(drop=True)
    )

    val_contexts["is_seen"] = val_contexts["search_query_norm"].isin(train_norm_set)
    val_contexts["is_unseen"] = ~val_contexts["is_seen"]

    n_val = len(val_contexts)
    n_seen = int(val_contexts["is_seen"].sum())
    n_unseen = int(val_contexts["is_unseen"].sum())

    print(f"Validation total contexts: {n_val}")
    print(f"  Seen:   {n_seen} ({n_seen / n_val:.2%})")
    print(f"  Unseen: {n_unseen} ({n_unseen / n_val:.2%})")

    # Check zero overlap for unseen
    unseen_queries = set(
        val_contexts.loc[val_contexts["is_unseen"], "search_query_norm"]
    )
    overlap = unseen_queries & train_norm_set
    assert len(overlap) == 0, f"Error: {len(overlap)} unseen queries overlap with train!"
    print(
        f"Strict isolation verified: 0 query text overlap between train and unseen validation."
    )

    # Build ground truth map
    gt_df = val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates()
    relevant = ground_truth_map(gt_df)
    relevant_json = {qid: list(items) for qid, items in relevant.items()}

    # Compute location coordinates from train
    loc_geo = (
        train.dropna(subset=["item_latitude", "item_longitude"])
        .groupby("item_location_id")[["item_latitude", "item_longitude"]]
        .mean()
    )
    loc_coords = {
        int(loc): {
            "lat": float(row["item_latitude"]),
            "lon": float(row["item_longitude"]),
        }
        for loc, row in loc_geo.iterrows()
    }

    # Save artifacts
    val_contexts.to_parquet(out_dir / "val_contexts.parquet", index=False)
    cols_to_save = [
        "internal_query_id",
        "search_query",
        "search_query_norm",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
        "item_id",
        "item_title_raw",
        "item_infm_params_text",
        "item_description_raw",
        "item_microcat_id",
        "item_category_id",
        "item_location_id",
        "item_latitude",
        "item_longitude",
        "fold",
    ]
    train_part[cols_to_save].to_parquet(
        out_dir / "train_part.parquet",
        index=False,
    )

    with (out_dir / "ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump(relevant_json, f)

    with (out_dir / "loc_coords.json").open("w", encoding="utf-8") as f:
        json.dump(loc_coords, f)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_train_rows": len(train),
        "train_part_rows": len(train_part),
        "val_contexts_count": n_val,
        "val_seen_count": n_seen,
        "val_unseen_count": n_unseen,
        "unseen_fraction": n_unseen / n_val,
        "benchmark_proxy_weights": {"seen": 0.35, "unseen": 0.65},
    }
    with (out_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved to {out_dir} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
