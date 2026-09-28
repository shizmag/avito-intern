"""Candidate pool union, fusion, evaluation contract, and diagnostic tools."""

import math
from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = [
    "assert_recall_monotonic",
    "evaluate_rankings",
    "evaluate_slices",
    "incremental_union_coverage",
    "oracle_candidate_coverage",
    "reciprocal_rank_fusion",
]


def assert_recall_monotonic(metrics: dict[str, float]) -> None:
    """Validate recall@k is monotonic non-decreasing with tolerance 1e-9.

    Validates recall@50 <= recall@100 <= recall@200 <= recall@500 <= recall@1000
    (and any other recall@k cutoffs present in metrics).

    Raises:
        AssertionError: If recall@k_prev > recall@k_next + 1e-9 for any k_prev < k_next.
    """
    ks: list[int] = []
    for key in metrics:
        if key.startswith("recall@"):
            suffix = key[len("recall@") :]
            try:
                ks.append(int(suffix))
            except ValueError:
                continue

    ks.sort()
    for i in range(len(ks) - 1):
        k_prev = ks[i]
        k_next = ks[i + 1]
        val_prev = metrics[f"recall@{k_prev}"]
        val_next = metrics[f"recall@{k_next}"]
        if val_prev - val_next > 1e-9:
            raise AssertionError(
                f"Recall monotonicity violated: recall@{k_prev} ({val_prev:.9f}) > recall@{k_next} ({val_next:.9f})"
            )


def evaluate_rankings(
    rankings: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    relevant: dict[str, set[str]],
    ks: Sequence[int] = (50, 100, 200, 500, 1000),
) -> dict[str, float]:
    """Calculate recall@k for each cutoff in ks and validate monotonicity.

    Parameters:
        rankings: Sequence of ranked candidates (item_id, score) per query.
        query_ids: Sequence of query IDs matching rankings.
        relevant: Ground-truth relevant item IDs mapped by query ID.
        ks: Sequence of cutoffs to evaluate.

    Returns:
        Dictionary mapping recall@k to recall value.

    Raises:
        ValueError: If rankings and query_ids length mismatch or any k < 1.
        AssertionError: If recall monotonicity is violated.
    """
    if len(rankings) != len(query_ids):
        raise ValueError(
            f"rankings and query_ids length mismatch: {len(rankings)} vs {len(query_ids)}"
        )
    for k in ks:
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")

    unique_items_per_query: list[list[str]] = []
    for ranking in rankings:
        seen: set[str] = set()
        items: list[str] = []
        for item_id, _ in ranking:
            if item_id not in seen:
                seen.add(item_id)
                items.append(item_id)
        unique_items_per_query.append(items)

    num_queries = len(query_ids)
    metrics: dict[str, float] = {}

    for k in ks:
        if num_queries == 0:
            metrics[f"recall@{k}"] = 0.0
            continue

        q_recalls: list[float] = []
        for q_idx, qid in enumerate(query_ids):
            rel = relevant.get(qid, set())
            if not rel:
                q_recalls.append(0.0)
                continue
            top_k = set(unique_items_per_query[q_idx][:k])
            q_recalls.append(len(top_k & rel) / len(rel))

        metrics[f"recall@{k}"] = sum(q_recalls) / num_queries

    assert_recall_monotonic(metrics)
    return metrics


def oracle_candidate_coverage(
    sources: Sequence[Sequence[Sequence[tuple[str, float]]]],
    query_ids: Sequence[str],
    relevant: dict[str, set[str]],
    ks: Sequence[int] = (50, 100, 200, 500, 1000),
) -> dict[str, float]:
    """Measure true top-K-per-source union coverage without post-union truncation.

    For each query and each cutoff k, takes the top-k unique candidates from each
    source, forms their unconstrained union, and computes recall against ground-truth.

    Parameters:
        sources: Sequence of sources, each containing candidate lists per query.
        query_ids: Sequence of query IDs.
        relevant: Ground-truth relevant item IDs mapped by query ID.
        ks: Sequence of cutoffs to evaluate.

    Returns:
        Dictionary mapping union_coverage@k to coverage value.
    """
    num_queries = len(query_ids)
    for s_idx, source in enumerate(sources):
        if len(source) != num_queries:
            raise ValueError(
                f"Source {s_idx} length ({len(source)}) does not match query_ids length ({num_queries})"
            )
    for k in ks:
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")

    # Extract unique items in order per source and query
    source_unique_items: list[list[list[str]]] = []
    for q_idx in range(num_queries):
        q_sources: list[list[str]] = []
        for source in sources:
            seen: set[str] = set()
            items: list[str] = []
            for item_id, _ in source[q_idx]:
                if item_id not in seen:
                    seen.add(item_id)
                    items.append(item_id)
            q_sources.append(items)
        source_unique_items.append(q_sources)

    metrics: dict[str, float] = {}
    for k in ks:
        if num_queries == 0 or len(sources) == 0:
            metrics[f"union_coverage@{k}"] = 0.0
            continue

        q_coverages: list[float] = []
        for q_idx, qid in enumerate(query_ids):
            rel = relevant.get(qid, set())
            if not rel:
                q_coverages.append(0.0)
                continue

            union_pool: set[str] = set()
            for items in source_unique_items[q_idx]:
                union_pool.update(items[:k])

            q_coverages.append(len(union_pool & rel) / len(rel))

        metrics[f"union_coverage@{k}"] = sum(q_coverages) / num_queries

    return metrics


def reciprocal_rank_fusion(
    sources: Sequence[Sequence[Sequence[tuple[str, float]]]],
    weights: Sequence[float] | None = None,
    c: float = 60.0,
    limit: int = 50,
) -> list[list[tuple[str, float]]]:
    """Weighted Reciprocal Rank Fusion combiner across candidate lists for each query.

    Stable tie-breaking (score descending, item_id ascending) with deduplication.

    Parameters:
        sources: Sequence of sources, each containing candidate lists per query.
        weights: Optional sequence of source weights. Defaults to 1.0 for each source.
        c: RRF smoothing constant (default 60.0).
        limit: Maximum number of fused candidates to retain per query (default 50).

    Returns:
        List of fused rankings (item_id, score) per query.
    """
    if len(sources) == 0:
        return []
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    num_sources = len(sources)
    num_queries = len(sources[0])
    for s_idx, source in enumerate(sources):
        if len(source) != num_queries:
            raise ValueError(
                f"Source {s_idx} length ({len(source)}) does not match expected ({num_queries})"
            )

    if weights is not None:
        if len(weights) != num_sources:
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match number of sources ({num_sources})"
            )
        w = [float(x) for x in weights]
    else:
        w = [1.0] * num_sources

    fused_rankings: list[list[tuple[str, float]]] = []

    for q_idx in range(num_queries):
        if limit == 0:
            fused_rankings.append([])
            continue

        scores: dict[str, float] = {}
        for s_idx, source in enumerate(sources):
            weight = w[s_idx]
            seen_in_source: set[str] = set()
            rank = 1
            for item_id, _ in source[q_idx]:
                if item_id not in seen_in_source:
                    seen_in_source.add(item_id)
                    scores[item_id] = scores.get(item_id, 0.0) + weight / (c + rank)
                    rank += 1

        # Stable tie-breaking: score desc, item_id asc
        sorted_candidates = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        fused_rankings.append(sorted_candidates[:limit])

    return fused_rankings


def incremental_union_coverage(
    base_sources: Sequence[Sequence[Sequence[tuple[str, float]]]],
    new_source: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    relevant: dict[str, set[str]],
    k: int = 500,
) -> float:
    """Historical intent incremental coverage check: coverage(base U new) - coverage(base).

    Parameters:
        base_sources: Sequence of base sources.
        new_source: Candidate list per query from the new candidate branch.
        query_ids: Sequence of query IDs.
        relevant: Ground-truth relevant item IDs mapped by query ID.
        k: Cutoff to evaluate coverage (default 500).

    Returns:
        Delta coverage: coverage(base U new) - coverage(base).
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if len(new_source) != len(query_ids):
        raise ValueError(
            f"new_source length ({len(new_source)}) does not match query_ids length ({len(query_ids)})"
        )
    for s_idx, source in enumerate(base_sources):
        if len(source) != len(query_ids):
            raise ValueError(
                f"Base source {s_idx} length ({len(source)}) does not match query_ids length ({len(query_ids)})"
            )

    key = f"union_coverage@{k}"
    if len(base_sources) == 0:
        cov_base = 0.0
    else:
        cov_base = oracle_candidate_coverage(
            base_sources, query_ids, relevant, ks=(k,)
        )[key]

    combined_sources = list(base_sources) + [new_source]
    cov_combined = oracle_candidate_coverage(
        combined_sources, query_ids, relevant, ks=(k,)
    )[key]

    return max(0.0, cov_combined - cov_base)


def _safe_bool(val: object) -> bool:
    if val is None or val is pd.NA:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    return bool(val)


def _align_slice_to_query_ids(
    slice_input: pd.Series | pd.DataFrame,
    query_ids: Sequence[str],
    query_df: pd.DataFrame,
) -> list[bool]:
    """Align a slice series to query_ids producing a list of booleans."""
    slice_series: pd.Series = (
        slice_input.iloc[:, 0] if isinstance(slice_input, pd.DataFrame) else slice_input
    )
    num_queries = len(query_ids)
    query_id_set = set(query_ids)
    index_as_str = [str(x) for x in slice_series.index]

    if any(idx in query_id_set for idx in index_as_str):
        mapping = {str(k): _safe_bool(v) for k, v in slice_series.items()}
        return [bool(mapping.get(qid, False)) for qid in query_ids]

    for col in ("internal_query_id", "query_id", "qid"):
        if col in query_df.columns:
            id_col = query_df[col].astype(str).tolist()
            if len(id_col) == len(slice_series):
                mapping = {
                    qid_val: _safe_bool(val)
                    for qid_val, val in zip(id_col, slice_series, strict=True)
                }
                return [bool(mapping.get(qid, False)) for qid in query_ids]

    if len(slice_series) == num_queries:
        return [_safe_bool(v) for v in slice_series]

    raise ValueError(
        f"Unable to align slice series of length {len(slice_series)} to {num_queries} queries"
    )


def evaluate_slices(
    rankings: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    query_df: pd.DataFrame,
    relevant: dict[str, set[str]],
    slices: Mapping[str, pd.Series | pd.DataFrame] | dict[str, pd.Series],
    ks: Sequence[int] = (50, 100, 200, 500, 1000),
) -> dict[str, dict[str, float]]:
    """Sliced metrics calculator.

    Parameters:
        rankings: Sequence of ranked candidates per query.
        query_ids: Sequence of query IDs matching rankings.
        query_df: DataFrame containing query metadata.
        relevant: Ground-truth relevant item IDs mapped by query ID.
        slices: Dictionary mapping slice names to boolean or categorical pd.Series.
        ks: Sequence of cutoffs for evaluation (default (50, 100, 200, 500, 1000)).

    Returns:
        Dictionary mapping slice names to dictionary of recall metrics.
    """
    if len(rankings) != len(query_ids):
        raise ValueError(
            f"rankings and query_ids length mismatch: {len(rankings)} vs {len(query_ids)}"
        )

    results: dict[str, dict[str, float]] = {}

    for slice_name, slice_input in slices.items():
        slice_series: pd.Series = (
            slice_input.iloc[:, 0]
            if isinstance(slice_input, pd.DataFrame)
            else slice_input
        )
        is_bool = pd.api.types.is_bool_dtype(slice_series) or set(
            slice_series.dropna().unique()
        ).issubset({True, False})
        if is_bool:
            mask = _align_slice_to_query_ids(slice_series, query_ids, query_df)
            slice_rankings = [r for r, m in zip(rankings, mask, strict=True) if m]
            slice_query_ids = [q for q, m in zip(query_ids, mask, strict=True) if m]
            if not slice_query_ids:
                results[slice_name] = {f"recall@{k}": 0.0 for k in ks}
            else:
                results[slice_name] = evaluate_rankings(
                    slice_rankings, slice_query_ids, relevant, ks=ks
                )
        else:
            for cat_val in slice_series.dropna().unique():
                cat_mask_series = slice_series == cat_val
                cat_name = f"{slice_name}_{cat_val}"
                mask = _align_slice_to_query_ids(cat_mask_series, query_ids, query_df)
                slice_rankings = [r for r, m in zip(rankings, mask, strict=True) if m]
                slice_query_ids = [q for q, m in zip(query_ids, mask, strict=True) if m]
                if not slice_query_ids:
                    results[cat_name] = {f"recall@{k}": 0.0 for k in ks}
                else:
                    results[cat_name] = evaluate_rankings(
                        slice_rankings, slice_query_ids, relevant, ks=ks
                    )

    return results
