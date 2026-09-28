"""Reciprocal-rank fusion."""

from __future__ import annotations

import pandas as pd

from ..candidates import validate_candidates


def reciprocal_rank_fusion(
    candidates: pd.DataFrame, *, rrf_k: int = 60, limit: int | None = None
) -> pd.DataFrame:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    validate_candidates(candidates)
    work = candidates.copy()
    work = work.assign(rrf_component=1.0 / (rrf_k + work["rank"]))
    result = work.groupby(["internal_query_id", "item_id"], as_index=False).agg(
        rrf_score=("rrf_component", "sum"), retriever_count=("source", "nunique")
    )
    result = result.assign(source="rrf", score=result["rrf_score"])
    result = result.sort_values(
        ["internal_query_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )  # pyright: ignore[reportCallIssue]
    result["rank"] = result.groupby("internal_query_id").cumcount() + 1
    if limit is not None:
        result = result.groupby("internal_query_id", group_keys=False).head(limit)
        result["rank"] = result.groupby("internal_query_id").cumcount() + 1
    return pd.DataFrame(
        result[
            [
                "internal_query_id",
                "item_id",
                "source",
                "score",
                "rank",
                "rrf_score",
                "retriever_count",
            ]
        ].reset_index(drop=True)
    )
