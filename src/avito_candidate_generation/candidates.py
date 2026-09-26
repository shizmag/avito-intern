"""Canonical candidate table helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = ("internal_query_id", "item_id", "source", "score", "rank")


class CandidateError(ValueError):
    """Candidate table contract violation."""


def candidate_frame(rows: Iterable[tuple[str, str, str, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=CANDIDATE_COLUMNS)
    validate_candidates(frame)
    return frame


def validate_candidates(
    frame: pd.DataFrame,
    *,
    known_item_ids: set[str] | None = None,
    allow_empty: bool = True,
) -> None:
    missing = set(CANDIDATE_COLUMNS).difference(frame.columns)
    if missing:
        raise CandidateError(f"missing candidate columns: {sorted(missing)}")
    if not allow_empty and frame.empty:
        raise CandidateError("candidate table is empty")
    for column in ("internal_query_id", "item_id", "source"):
        if not pd.api.types.is_string_dtype(frame[column]):
            raise CandidateError(f"{column} must be string")
    scores = np.asarray(
        pd.to_numeric(frame["score"], errors="coerce"), dtype=np.float64
    )
    if not np.isfinite(scores).all():
        raise CandidateError("candidate scores must be finite")
    if (frame["rank"] < 1).any():
        raise CandidateError("rank must be positive")
    if frame.duplicated(["internal_query_id", "item_id", "source"]).any():
        raise CandidateError("duplicate candidate pair")
    if known_item_ids is not None and not set(frame["item_id"]).issubset(
        known_item_ids
    ):
        raise CandidateError("unknown item ID")
    try:
        groups = frame.groupby(["internal_query_id", "source"], sort=False)
    except (KeyError, TypeError) as exc:
        raise CandidateError("invalid candidate grouping columns") from exc
    for _, group in groups:
        try:
            ranks = sorted(int(x) for x in group["rank"])
        except (TypeError, ValueError, KeyError) as exc:
            raise CandidateError("rank values must be integers") from exc
        if ranks != list(range(1, len(ranks) + 1)):
            raise CandidateError("rank must be contiguous per query/source")


def rank_candidates(
    frame: pd.DataFrame, *, score_column: str = "score", limit: int = 50
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be positive")
    work = frame.sort_values(
        ["internal_query_id", score_column, "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work = work.drop_duplicates(["internal_query_id", "item_id"], keep="first")
    return work.groupby("internal_query_id", group_keys=False).head(limit).copy()
