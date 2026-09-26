"""Pair-level source and safe feature construction."""
from __future__ import annotations

import pandas as pd


def build_pair_features(candidates: pd.DataFrame, *, source_names: list[str] | None = None) -> pd.DataFrame:
    source_names = source_names or sorted(candidates["source"].astype(str).unique().tolist())
    rows: list[pd.DataFrame] = []
    for source in source_names:
        part = candidates[candidates["source"] == source][["internal_query_id", "item_id", "score", "rank"]].copy()
        part = part.rename(columns={"score": f"{source}_score", "rank": f"{source}_rank"})  # pyright: ignore[reportCallIssue,reportAttributeAccessIssue]
        part[f"{source}_present"] = True
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["internal_query_id", "item_id"])
    result = rows[0]
    for part in rows[1:]:
        result = result.merge(part, on=["internal_query_id", "item_id"], how="outer", validate="one_to_one")
    present = [f"{source}_present" for source in source_names]
    result["retriever_count"] = result[present].fillna(False).sum(axis=1)
    return result.sort_values(["internal_query_id", "item_id"], kind="mergesort").reset_index(drop=True)
