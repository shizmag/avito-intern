"""Exact batched dense similarity search."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
import pandas as pd


def _stable_top_indices(
    scores: np.ndarray, item_ids: Sequence[str], k: int
) -> list[int]:
    """Return exact top-k indices with score-desc/item-id-asc tie-breaking."""
    if k < 1 or len(scores) == 0:
        return []
    try:
        if len(scores) <= k:
            candidates = np.arange(len(scores), dtype=np.int64)
        else:
            boundary_index = int(np.argpartition(scores, -k)[-k])
            boundary = float(scores[boundary_index])
            candidates = np.flatnonzero(scores >= boundary)
        return sorted(
            (int(index) for index in candidates),
            key=lambda index: (-float(scores[index]), str(item_ids[index])),
        )[:k]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("invalid dense similarity scores") from exc


def _validate_inputs(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: Sequence[str],
    k: int,
    batch_size: int,
    item_batch_size: int,
    query_ids: Sequence[str] | None,
    source: str,
) -> list[str]:
    if (
        query_embeddings.ndim != 2
        or item_embeddings.ndim != 2
        or query_embeddings.shape[1] != item_embeddings.shape[1]
    ):
        raise ValueError("embeddings must be 2D with equal dimensions")
    if len(item_ids) != item_embeddings.shape[0]:
        raise ValueError("item_ids length mismatch")
    if k < 1 or batch_size < 1 or item_batch_size < 1:
        raise ValueError("k and batch sizes must be positive")
    if item_embeddings.shape[0] == 0:
        raise ValueError("item embeddings must be non-empty")
    if (
        not np.isfinite(query_embeddings).all()
        or not np.isfinite(item_embeddings).all()
    ):
        raise ValueError("embeddings must be finite")
    ids = (
        list(query_ids)
        if query_ids is not None
        else [str(i) for i in range(len(query_embeddings))]
    )
    if len(ids) != len(query_embeddings):
        raise ValueError("query_ids length mismatch")
    if not source:
        raise ValueError("source must be non-empty")
    if len({str(item_id) for item_id in item_ids}) != len(item_ids):
        raise ValueError("item_ids must be unique")
    return [str(query_id) for query_id in ids]


def iter_exact_top_k(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: Sequence[str],
    *,
    k: int = 50,
    query_ids: Sequence[str] | None = None,
    source: str = "dense",
    batch_size: int = 64,
    item_batch_size: int = 4096,
) -> Iterator[pd.DataFrame]:
    """Yield exact top-k frames per query batch without full similarity materialization."""
    ids = _validate_inputs(
        query_embeddings,
        item_embeddings,
        item_ids,
        k,
        batch_size,
        item_batch_size,
        query_ids,
        source,
    )
    for start in range(0, len(query_embeddings), batch_size):
        query_batch = query_embeddings[start : start + batch_size]
        candidates: list[list[tuple[int, float]]] = [
            [] for _ in range(len(query_batch))
        ]
        for item_start in range(0, len(item_embeddings), item_batch_size):
            item_end = min(item_start + item_batch_size, len(item_embeddings))
            scores_batch = query_batch @ item_embeddings[item_start:item_end].T
            if not np.isfinite(scores_batch).all():
                raise ValueError("invalid dense similarity scores")
            local_ids = item_ids[item_start:item_end]
            local_k = min(k, len(local_ids))
            for offset, scores in enumerate(scores_batch):
                local_order = _stable_top_indices(scores, local_ids, local_k)
                try:
                    candidates[offset].extend(
                        (item_start + index, float(scores[index]))
                        for index in local_order
                    )
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("invalid dense similarity scores") from exc
                candidates[offset] = sorted(
                    candidates[offset],
                    key=lambda pair: (-pair[1], str(item_ids[pair[0]])),
                )[:k]
        rows: list[tuple[str, str, float, int]] = []
        for offset, pairs in enumerate(candidates):
            rows.extend(
                (ids[start + offset], str(item_ids[index]), score, rank)
                for rank, (index, score) in enumerate(pairs, 1)
            )
        result = pd.DataFrame(
            rows, columns=["internal_query_id", "item_id", "score", "rank"]
        )
        result["source"] = source
        output = cast(
            pd.DataFrame,
            result.loc[
                :, ["internal_query_id", "item_id", "source", "score", "rank"]
            ].copy(),
        )
        yield output


def exact_top_k(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: list[str],
    *,
    k: int = 50,
    query_ids: list[str] | None = None,
    source: str = "dense",
    batch_size: int = 64,
    item_batch_size: int = 4096,
) -> pd.DataFrame:
    """Return exact top-k results; similarity memory is bounded by both batch sizes."""
    frames = list(
        iter_exact_top_k(
            query_embeddings,
            item_embeddings,
            item_ids,
            k=k,
            query_ids=query_ids,
            source=source,
            batch_size=batch_size,
            item_batch_size=item_batch_size,
        )
    )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["internal_query_id", "item_id", "source", "score", "rank"]
        )
    )
