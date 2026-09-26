"""Exact batched dense similarity search."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exact_top_k(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: list[str],
    *,
    k: int = 50,
    query_ids: list[str] | None = None,
) -> pd.DataFrame:
    if (
        query_embeddings.ndim != 2
        or item_embeddings.ndim != 2
        or query_embeddings.shape[1] != item_embeddings.shape[1]
    ):
        raise ValueError("embeddings must be 2D with equal dimensions")
    if len(item_ids) != item_embeddings.shape[0]:
        raise ValueError("item_ids length mismatch")
    if k < 1:
        raise ValueError("k must be positive")
    ids = (
        query_ids
        if query_ids is not None
        else [str(i) for i in range(len(query_embeddings))]
    )
    rows: list[tuple[str, str, float, int]] = []
    for qi, vector in enumerate(query_embeddings):
        scores = item_embeddings @ vector
        order = sorted(
            range(len(item_ids)), key=lambda i: (-float(scores[i]), item_ids[i])
        )[:k]
        rows.extend(
            (ids[qi], item_ids[i], float(scores[i]), rank)
            for rank, i in enumerate(order, 1)
        )
    result = pd.DataFrame(
        rows, columns=["internal_query_id", "item_id", "score", "rank"]
    )
    result["source"] = "dense"
    return pd.DataFrame(result[["internal_query_id", "item_id", "source", "score", "rank"]])
