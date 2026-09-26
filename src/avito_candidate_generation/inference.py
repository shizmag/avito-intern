"""Benchmark prediction schema and deterministic top-50 inference helper."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def rank_predictions(candidates: pd.DataFrame, *, limit: int = 50) -> pd.DataFrame:
    required = {"query_id", "item_id", "final_score"}
    if not required.issubset(candidates.columns):
        raise ValueError("inference candidates require query_id, item_id, final_score")
    if limit < 1:
        raise ValueError("limit must be positive")
    work = candidates.copy()
    if "rrf_score" not in work:
        work["rrf_score"] = 0.0
    work = work.sort_values(
        ["query_id", "final_score", "rrf_score", "item_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )  # pyright: ignore[reportCallIssue]
    work = work.drop_duplicates(["query_id", "item_id"], keep="first")
    work = work.groupby("query_id", group_keys=False).head(limit).copy()
    work["rank"] = work.groupby("query_id").cumcount() + 1
    result = work[["query_id", "item_id", "rank", "final_score", "rrf_score"]].reset_index(drop=True)
    return pd.DataFrame(result)


def write_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target
