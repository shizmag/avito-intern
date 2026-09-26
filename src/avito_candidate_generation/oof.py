"""Out-of-fold candidate feature table assembly."""
from __future__ import annotations

import pandas as pd


def build_oof_table(features: pd.DataFrame, ground_truth: pd.DataFrame, *, fold: int) -> pd.DataFrame:
    labels = ground_truth[["internal_query_id", "item_id"]].drop_duplicates().assign(label=1)
    result = features.merge(labels, on=["internal_query_id", "item_id"], how="left", validate="one_to_one")
    result["label"] = result["label"].fillna(0).astype("int8")
    result["fold"] = fold
    return result.sort_values(["internal_query_id", "item_id"], kind="mergesort").reset_index(drop=True)
