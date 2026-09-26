"""Leakage-safe sampled hard-negative construction."""
from __future__ import annotations

import pandas as pd


def mine_hard_negatives(candidates: pd.DataFrame, known_positive: pd.DataFrame, *, max_per_query: int = 20) -> pd.DataFrame:
    required = {"internal_query_id", "item_id", "source", "rank", "score"}
    if not required.issubset(candidates.columns):
        raise ValueError("candidate columns missing")
    positive_pairs = set(zip(known_positive["internal_query_id"].astype(str), known_positive["item_id"].astype(str)))
    work = candidates.copy()
    work["_pair"] = list(zip(work["internal_query_id"].astype(str), work["item_id"].astype(str)))
    work = work[~work["_pair"].isin(list(positive_pairs))].drop(columns="_pair")  # pyright: ignore[reportArgumentType]
    work = work.sort_values(["internal_query_id", "rank", "item_id"], kind="mergesort")  # pyright: ignore[reportCallIssue]
    work["iteration"] = 0
    work["sampling_order"] = work.groupby("internal_query_id").cumcount() + 1
    work["label"] = 0
    return work.groupby("internal_query_id", group_keys=False).head(max_per_query).reset_index(drop=True)
