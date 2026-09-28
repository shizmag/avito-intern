"""Research-stage, leakage-aware retrieval helpers for Avito candidate generation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import canonical_fingerprint, stable_internal_query_id

CONTEXT_COLUMNS: tuple[str, ...] = (
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
)
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "internal_query_id",
    "item_id",
    "source",
    "score",
    "rank",
)
_NUMBER_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?(?:кг|г|см|мм|м|л|мл|вт|квм|кв|год(?:а|ы)?|лет|ч|час(?:а|ов)?))?(?!\w)",
    re.IGNORECASE,
)


def normalize_research_text(value: object) -> str:
    """Apply deterministic lightweight normalization; preserve useful numbers."""
    if value is None:
        return ""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(text.split())


def context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one stable ID per complete search context and reject inconsistent IDs."""
    missing = sorted(set(CONTEXT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing context columns: {missing}")
    work = frame.copy()
    records = [
        dict(zip(CONTEXT_COLUMNS, row, strict=True))
        for row in work[list(CONTEXT_COLUMNS)]
        .fillna("<NULL>")
        .itertuples(index=False, name=None)
    ]
    work["internal_query_id"] = [
        stable_internal_query_id(record, list(CONTEXT_COLUMNS)) for record in records
    ]
    return work


def context_folds(
    context_ids: Iterable[str], *, seed: int = 42, folds: int = 5
) -> pd.DataFrame:
    """Assign complete contexts to deterministic hash buckets."""
    if folds < 2 or seed < 0:
        raise ValueError("folds must be >= 2 and seed must be non-negative")
    unique = sorted({str(value) for value in context_ids})
    bucket = [
        int.from_bytes(
            hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big"
        )
        % folds
        for value in unique
    ]
    return pd.DataFrame({"internal_query_id": unique, "fold": bucket})


@dataclass(frozen=True)
class ResearchValidation:
    """Benchmark-universe validation built from held-out full search contexts."""

    train_contexts: pd.DataFrame
    validation_contexts: pd.DataFrame
    ground_truth: pd.DataFrame
    context_folds: pd.DataFrame
    benchmark_items: pd.DataFrame
    protocol: dict[str, Any]


def build_research_validation(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    *,
    seed: int = 42,
    train_fold: int = 0,
    folds: int = 5,
) -> ResearchValidation:
    """Build OOF context validation over the complete benchmark item universe.

    Only positive interactions from held-out contexts that are present in the
    benchmark corpus become ground truth. Unknown interactions remain unknown;
    no unobserved pair is converted into a negative label.
    """
    required_train = set(CONTEXT_COLUMNS) | {"item_id", "item_microcat_id"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train missing {sorted(required_train.difference(train.columns))}")
    required_items = {"item_id", "item_microcat_id"}
    if not required_items.issubset(benchmark_items.columns):
        raise ValueError(
            f"benchmark_items missing {sorted(required_items.difference(benchmark_items.columns))}"
        )
    if benchmark_items["item_id"].duplicated().any():
        raise ValueError("benchmark item IDs must be unique")
    if train_fold < 0 or train_fold >= folds:
        raise ValueError("train_fold must be within folds")
    if folds < 2:
        raise ValueError("folds must be at least 2")

    work: Any = context_frame(train)
    folds_frame: Any = context_folds(
        work["internal_query_id"].astype(str), seed=seed, folds=folds
    )
    context_columns = [*CONTEXT_COLUMNS, "internal_query_id"]
    contexts: Any = work[context_columns].drop_duplicates()
    contexts = pd.merge(
        contexts,
        folds_frame,
        on="internal_query_id",
        how="left",
        validate="one_to_one",
    )
    if bool(contexts["fold"].isna().any()):
        raise ValueError("context fold assignment incomplete")

    benchmark_ids = set(benchmark_items["item_id"].astype(str))
    positives: Any = work[["internal_query_id", "item_id", "item_microcat_id"]].copy()
    positives["internal_query_id"] = positives["internal_query_id"].astype(str)
    positives["item_id"] = positives["item_id"].astype(str)
    positives = positives.drop_duplicates()
    positives = cast(Any, positives[
        positives["item_id"].isin(list(benchmark_ids))
    ].copy())
    positives = cast(Any, positives.copy())
    positives["train_item_microcat_id"] = cast(Any, positives.pop("item_microcat_id"))
    benchmark_microcats = benchmark_items[["item_id", "item_microcat_id"]].copy()
    benchmark_microcats.columns = ["item_id", "benchmark_item_microcat_id"]
    positives = cast(Any, pd.merge(
        cast(pd.DataFrame, positives),
        benchmark_microcats,
        on="item_id",
        how="left",
        validate="many_to_one",
    ))
    if bool(positives["benchmark_item_microcat_id"].isna().any()):
        raise ValueError("benchmark positive has missing microcategory")

    validation_contexts: Any = contexts[contexts["fold"] != train_fold].copy()
    positive_counts: Any = pd.DataFrame(
        {
            "internal_query_id": positives["internal_query_id"].drop_duplicates(),
            "known_positive_count": positives.groupby("internal_query_id")[
                "item_id"
            ].nunique().reindex(
                positives["internal_query_id"].drop_duplicates()
            ).to_numpy(),
        }
    )
    validation_contexts = pd.merge(
        cast(pd.DataFrame, validation_contexts),
        cast(pd.DataFrame, positive_counts),
        on="internal_query_id",
        how="inner",
        validate="one_to_one",
    )
    validation_ids = set(validation_contexts["internal_query_id"].astype(str))
    ground_truth: Any = positives[
        positives["internal_query_id"].isin(list(validation_ids))
    ][["internal_query_id", "item_id"]].drop_duplicates()
    train_contexts: Any = contexts[contexts["fold"] == train_fold].copy()
    train_context_ids = set(train_contexts["internal_query_id"].astype(str))
    train_positive_rows = len(
        positives[positives["internal_query_id"].isin(list(train_context_ids))]
    )
    protocol = {
        "schema_version": 2,
        "protocol": "context-holdout-over-full-benchmark-corpus",
        "seed": seed,
        "folds": folds,
        "train_fold": train_fold,
        "context_columns": list(CONTEXT_COLUMNS),
        "unknown_interactions_are_not_negatives": True,
        "train_contexts": len(train_contexts),
        "validation_contexts_with_known_benchmark_positive": len(validation_contexts),
        "validation_ground_truth_pairs": len(ground_truth),
        "benchmark_items": len(benchmark_items),
        "train_positive_rows_in_train_fold": train_positive_rows,
        "fingerprints": {
            "train": canonical_fingerprint(
            cast(pd.DataFrame, train[list(CONTEXT_COLUMNS) + ["item_id"]])
        ),
            "benchmark_items": canonical_fingerprint(
                cast(pd.DataFrame, benchmark_items)
            ),
            "context_folds": canonical_fingerprint(
                cast(pd.DataFrame, folds_frame)
            ),
            "validation_contexts": canonical_fingerprint(
                cast(pd.DataFrame, validation_contexts)
            ),
            "ground_truth": canonical_fingerprint(
                cast(pd.DataFrame, ground_truth)
            ),
        },
    }
    return ResearchValidation(
        train_contexts=train_contexts.reset_index(drop=True),
        validation_contexts=validation_contexts.reset_index(drop=True),
        ground_truth=ground_truth.reset_index(drop=True),
        context_folds=folds_frame,
        benchmark_items=benchmark_items.reset_index(drop=True),
        protocol=protocol,
    )


def write_research_validation(
    validation: ResearchValidation, output_dir: str | Path
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "train_contexts": validation.train_contexts,
        "validation_contexts": validation.validation_contexts,
        "ground_truth": validation.ground_truth,
        "context_folds": validation.context_folds,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    protocol_path = out / "protocol.json"
    import json

    protocol_path.write_text(
        json.dumps(validation.protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["protocol"] = protocol_path
    return paths


def query_text(frame: pd.DataFrame, *, include_location: bool = True) -> list[str]:
    """Build structured historical-query text without using item-side fields."""
    required = set(CONTEXT_COLUMNS).difference(frame.columns)
    if required:
        raise ValueError(f"missing query fields: {sorted(required)}")
    result: list[str] = []
    records = [
        dict(zip(CONTEXT_COLUMNS, row, strict=True))
        for row in frame[list(CONTEXT_COLUMNS)].itertuples(index=False, name=None)
    ]
    for row in records:
        parts = [
            normalize_research_text(row["search_query"]),
            normalize_research_text(row["search_infm_params_text"]),
            f"category_{normalize_research_text(row['search_category'])}",
        ]
        if include_location:
            parts.append(f"location_{normalize_research_text(row['search_location_id'])}")
            parts.append(f"delivery_{normalize_research_text(row['search_is_delivery_search'])}")
        result.append(" ".join(part for part in parts if part))
    return result


def field_text(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing text fields: {missing}")
    values = frame[list(columns)].fillna("").astype(str)
    return [
        " ".join(normalize_research_text(value) for value in row)
        for row in values.itertuples(index=False, name=None)
    ]


def canonical_numbers(text: object) -> set[str]:
    value = normalize_research_text(text)
    return {
        re.sub(r"\s+", "", token.replace(",", "."))
        for token in _NUMBER_RE.findall(value)
    }


@dataclass
class SparseRetrievalIndex:
    """Small deterministic sparse cosine index used by research ablations."""

    item_ids: list[str]
    vectorizer: TfidfVectorizer
    matrix: Any
    name: str

    @classmethod
    def fit(
        cls,
        item_ids: Sequence[str],
        texts: Sequence[str],
        *,
        name: str,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int | None = None,
    ) -> SparseRetrievalIndex:
        if len(item_ids) != len(texts) or not item_ids:
            raise ValueError("item IDs and texts must be non-empty and equally sized")
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            min_df=min_df,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(list(texts)).tocsr()
        return cls([str(value) for value in item_ids], vectorizer, matrix, name)

    def top_k(self, texts: Sequence[str], *, k: int, batch_size: int = 64) -> list[list[tuple[str, float]]]:
        if k < 1 or batch_size < 1:
            raise ValueError("k and batch_size must be positive")
        if not texts:
            return []
        count = min(k, len(self.item_ids))
        all_ids = sorted(self.item_ids)
        result: list[list[tuple[str, float]]] = []
        for start in range(0, len(texts), batch_size):
            query_matrix = self.vectorizer.transform(
                list(texts[start : start + batch_size])
            )
            scores = (query_matrix @ self.matrix.T).toarray()
            for row in scores:
                if len(row) <= count:
                    indices = np.arange(len(row), dtype=np.int64)
                else:
                    indices = np.argpartition(row, -count)[-count:]
                ranked: list[tuple[str, float]] = []
                for index in indices:
                    try:
                        ranked.append(
                            (self.item_ids[int(index)], float(row[int(index)]))
                        )
                    except (IndexError, TypeError, ValueError) as exc:
                        raise ValueError("invalid sparse retrieval score") from exc
                ranked.sort(key=lambda pair: (-pair[1], pair[0]))
                seen = {item_id for item_id, _ in ranked}
                if len(ranked) < count:
                    ranked.extend((item_id, 0.0) for item_id in all_ids if item_id not in seen)
                result.append(ranked[:count])
        return result

    def top_k_routed(
        self,
        texts: Sequence[str],
        allowed_item_ids: Sequence[set[str]],
        *,
        k: int,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """Retrieve only from per-query allowed item IDs."""
        if len(texts) != len(allowed_item_ids):
            raise ValueError("texts and route sets must have equal length")
        if k < 1 or batch_size < 1:
            raise ValueError("k and batch_size must be positive")
        item_positions = {item_id: index for index, item_id in enumerate(self.item_ids)}
        allowed_indices = [
            np.asarray(
                sorted(
                    item_positions[item_id]
                    for item_id in item_set
                    if item_id in item_positions
                ),
                dtype=np.int64,
            )
            for item_set in allowed_item_ids
        ]
        result: list[list[tuple[str, float]]] = []
        for start in range(0, len(texts), batch_size):
            query_matrix = self.vectorizer.transform(
                list(texts[start : start + batch_size])
            )
            scores = (query_matrix @ self.matrix.T).toarray()
            for row, allowed in zip(
                scores, allowed_indices[start : start + batch_size], strict=True
            ):
                if len(allowed) == 0:
                    result.append([])
                    continue
                count = min(k, len(allowed))
                selected_scores = row[allowed]
                if len(selected_scores) <= count:
                    local = np.arange(len(selected_scores), dtype=np.int64)
                else:
                    local = np.argpartition(selected_scores, -count)[-count:]
                ranked: list[tuple[str, float]] = []
                for index in local:
                    try:
                        ranked.append(
                            (
                                self.item_ids[int(allowed[int(index)])],
                                float(selected_scores[int(index)]),
                            )
                        )
                    except (IndexError, TypeError, ValueError) as exc:
                        raise ValueError("invalid routed retrieval score") from exc
                ranked.sort(key=lambda pair: (-pair[1], pair[0]))
                result.append(ranked[:count])
        return result


def numeric_rank(
    queries: Sequence[str], item_texts: Sequence[str], item_ids: Sequence[str], *, k: int
) -> list[list[tuple[str, float]]]:
    """Retrieve by exact canonical number overlap, with deterministic ties."""
    inverted: dict[str, list[int]] = {}
    for index, text in enumerate(item_texts):
        for number in canonical_numbers(text):
            inverted.setdefault(number, []).append(index)
    output: list[list[tuple[str, float]]] = []
    for query in queries:
        query_numbers = canonical_numbers(query)
        counts: dict[int, int] = {}
        for number in query_numbers:
            for index in inverted.get(number, []):
                counts[index] = counts.get(index, 0) + 1
        ranked = sorted(
            ((str(item_ids[index]), count / max(len(query_numbers), 1)) for index, count in counts.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        output.append(ranked[:k])
    return output


def ground_truth_map(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    if ground_truth.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate ground-truth pair")
    return {
        str(query_id): set(group["item_id"].astype(str))
        for query_id, group in ground_truth.groupby("internal_query_id", sort=False)
    }


def ranking_metrics(
    rankings: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    relevant: dict[str, set[str]],
    ks: Sequence[int] = (50, 100, 200, 500, 1000),
) -> dict[str, float]:
    if len(rankings) != len(query_ids):
        raise ValueError("ranking/query length mismatch")
    if not relevant:
        raise ValueError("empty ground truth")
    try:
        unique_ks = tuple(dict.fromkeys(int(k) for k in ks))
        sums = dict.fromkeys(unique_ks, 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("ks must contain integers") from exc
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        if str(query_id) not in relevant:
            raise ValueError(f"missing ground truth for query {query_id}")
        expected = relevant[str(query_id)]
        ids = [str(item_id) for item_id, _ in ranking]
        for key in unique_ks:
            try:
                sums[key] += len(set(ids[:key]) & expected) / len(expected)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("invalid ranking metric input") from exc
    try:
        return {
            f"recall@{k}": sums[k] / len(query_ids) for k in unique_ks
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid ranking metric input") from exc


def union_rankings(
    rankings: Sequence[Sequence[Sequence[tuple[str, float]]]],
    *,
    weights: Sequence[float] | None = None,
    limit: int = 1000,
) -> list[list[tuple[str, float]]]:
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length mismatch")
    output: list[list[tuple[str, float]]] = []
    for per_query in zip(*rankings, strict=True):
        scores: dict[str, float] = {}
        for source_weight, ranking in zip(weights, per_query, strict=True):
            for rank, (item_id, _score) in enumerate(ranking, start=1):
                try:
                    scores[str(item_id)] = scores.get(str(item_id), 0.0) + float(source_weight) * (
                        1.0 / (60.0 + rank)
                    )
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise ValueError("invalid union score") from exc
        output.append(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit])
    return output


def write_candidate_frame(
    rankings: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    *,
    source: str,
) -> pd.DataFrame:
    rows: list[tuple[str, str, str, float, int]] = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        for rank, (item_id, score) in enumerate(ranking, start=1):
            try:
                rows.append((str(query_id), str(item_id), source, float(score), rank))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid candidate score") from exc
    return pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))


def oracle_union_metrics(
    rankings: Sequence[Sequence[Sequence[tuple[str, float]]]],
    query_ids: Sequence[str],
    relevant: dict[str, set[str]],
    ks: Sequence[int] = (500, 1000),
) -> dict[str, float]:
    """Measure true top-k-per-source union coverage without post-union truncation."""
    if not rankings:
        raise ValueError("rankings must not be empty")
    if len(query_ids) != len(next(iter(rankings))):
        raise ValueError("ranking/query length mismatch")
    try:
        values: dict[int, list[float]] = {int(k): [] for k in ks}
    except (TypeError, ValueError) as exc:
        raise ValueError("ks must contain integers") from exc
    for query_index, query_id in enumerate(query_ids):
        if str(query_id) not in relevant:
            raise ValueError(f"missing ground truth for query {query_id}")
        expected = relevant[str(query_id)]
        for k in ks:
            union: set[str] = set()
            for source in rankings:
                try:
                    union.update(
                        str(item_id)
                        for item_id, _ in source[query_index][: int(k)]
                    )
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("invalid union ranking") from exc
            try:
                values[int(k)].append(len(union & expected) / len(expected))
            except (KeyError, ZeroDivisionError) as exc:
                raise ValueError("invalid union metric input") from exc
    return {f"union_recall@{k}": sum(scores) / len(scores) for k, scores in values.items()}


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
