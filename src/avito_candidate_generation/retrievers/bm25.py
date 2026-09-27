"""Small dependency-free BM25 retriever for local deterministic baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing import tokenize


@dataclass
class BM25Index:
    item_ids: list[str]
    documents: list[tuple[str, ...]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.document_frequency: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, document in enumerate(self.documents):
            counts: dict[str, int] = {}
            for term in document:
                counts[term] = counts.get(term, 0) + 1
            for term, frequency in counts.items():
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
                self.postings.setdefault(term, []).append((index, frequency))
        self.document_lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.document_lengths) / max(1, len(self.documents))
        self._idf = {
            term: math.log(
                1.0 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in self.document_frequency.items()
        }
        self._item_order = sorted(
            range(len(self.item_ids)), key=self.item_ids.__getitem__
        )
        self._posting_indices = {
            term: np.asarray([index for index, _ in postings], dtype=np.int64)
            for term, postings in self.postings.items()
        }
        self._posting_contributions = {
            term: np.asarray(
                [
                    self._idf[term]
                    * (tf * (self.k1 + 1))
                    / (
                        tf
                        + self.k1
                        * (
                            1
                            - self.b
                            + self.b
                            * self.document_lengths[index]
                            / max(self.average_length, 1e-12)
                        )
                    )
                    for index, tf in postings
                ],
                dtype=np.float64,
            )
            for term, postings in self.postings.items()
        }

    @classmethod
    def fit(cls, items: pd.DataFrame, text_column: str = "text") -> BM25Index:
        return cls(
            items["item_id"].astype(str).tolist(),
            [tokenize(str(x) if pd.notna(x) else "") for x in items[text_column]],
        )

    def score(self, query: Sequence[str], document: tuple[str, ...]) -> float:
        counts = {term: document.count(term) for term in set(document)}
        result = 0.0
        for term in query:
            tf = counts.get(term)
            if tf is None:
                continue
            result += (
                self._idf[term]
                * (tf * (self.k1 + 1))
                / (
                    tf
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b * len(document) / max(self.average_length, 1e-12)
                    )
                )
            )
        return result

    def _retrieve_indices(
        self, tokens: Sequence[str], k: int
    ) -> list[tuple[int, float]]:
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        for term in tokens:
            indices = self._posting_indices.get(term)
            if indices is not None:
                scores[indices] += self._posting_contributions[term]
        matched = np.flatnonzero(scores > 0.0)
        try:
            if len(matched) > k:
                positive_scores = scores[matched]
                boundary_index = int(np.argpartition(positive_scores, -k)[-k])
                boundary = float(positive_scores[boundary_index])
                matched = matched[positive_scores >= boundary]
            ordered = sorted(
                (int(index) for index in matched),
                key=lambda index: (-float(scores[index]), self.item_ids[index]),
            )[:k]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("invalid BM25 scores") from exc
        if len(ordered) < k:
            matched_set = set(ordered)
            ordered.extend(
                index for index in self._item_order if index not in matched_set
            )
            ordered = ordered[:k]
        try:
            return [(index, float(scores[index])) for index in ordered]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("invalid BM25 scores") from exc

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_column: str = "text",
        k: int = 50,
        source: str = "bm25",
        query_batch_size: int = 256,
    ) -> pd.DataFrame:
        if k < 1 or query_batch_size < 1:
            raise ValueError("k and query_batch_size must be positive")
        rows: list[tuple[str, str, str, float, int]] = []
        query_rows = list(
            queries[[query_column, "internal_query_id"]].itertuples(
                index=False, name=None
            )
        )
        for start in range(0, len(query_rows), query_batch_size):
            batch = query_rows[start : start + query_batch_size]
            for value, query_id in batch:
                tokens = tokenize(str(value) if value is not None else "")
                order = self._retrieve_indices(tokens, min(k, len(self.item_ids)))
                rows.extend(
                    (
                        str(query_id),
                        self.item_ids[index],
                        source,
                        score,
                        rank,
                    )
                    for rank, (index, score) in enumerate(order, 1)
                )
        return pd.DataFrame(
            rows,
            columns=["internal_query_id", "item_id", "source", "score", "rank"],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".npz":
            np.savez_compressed(
                target,
                item_ids=np.asarray(self.item_ids, dtype=str),
                documents=np.asarray(
                    [" ".join(document) for document in self.documents], dtype=str
                ),
                k1=np.asarray(self.k1),
                b=np.asarray(self.b),
            )
        else:
            payload = {
                "item_ids": self.item_ids,
                "documents": [" ".join(document) for document in self.documents],
                "k1": self.k1,
                "b": self.b,
                "format": "tokens-v2",
            }
            target.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    @staticmethod
    def load(path: str | Path) -> BM25Index:
        target = Path(path)
        try:
            if target.suffix == ".npz":
                with np.load(target, allow_pickle=False) as payload:
                    item_ids = [str(value) for value in payload["item_ids"].tolist()]
                    documents = [
                        tuple(str(token) for token in str(value).split(" ") if token)
                        for value in payload["documents"].tolist()
                    ]
                    k1 = float(payload["k1"])
                    b = float(payload["b"])
            else:
                payload = json.loads(target.read_text(encoding="utf-8"))
                item_ids = payload.get("item_ids")
                documents = payload.get("documents")
                if not isinstance(item_ids, list) or not isinstance(documents, list):
                    raise ValueError("invalid BM25 index")
                documents = [
                    tuple(str(token) for token in document.split(" "))
                    if isinstance(document, str)
                    else tuple(str(token) for token in document)
                    for document in documents
                ]
                k1 = float(payload.get("k1", 1.5))
                b = float(payload.get("b", 0.75))
            if len(item_ids) != len(documents):
                raise ValueError("invalid BM25 index")
            return BM25Index(item_ids, documents, k1, b)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid BM25 index artifact") from exc


def retrieve_bm25(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    text_column: str = "text",
    query_column: str = "text",
    k: int = 50,
    source: str = "bm25",
    category_policy: str = "none",
    query_category_column: str = "search_category",
    item_category_column: str = "search_category",
) -> pd.DataFrame:
    if category_policy not in {"none", "hard", "fallback"}:
        raise ValueError("category_policy must be one of none, hard, fallback")
    index = BM25Index.fit(items, text_column)
    if category_policy == "none":
        return index.retrieve(queries, query_column, k, source)
    if query_category_column not in queries or item_category_column not in items:
        raise ValueError("category policy requires category columns on both tables")
    category_indices: dict[str, list[int]] = {}
    for index_number, value in enumerate(items[item_category_column].tolist()):
        category = "" if pd.isna(value) else str(value)
        if category and category != "nan":
            category_indices.setdefault(category, []).append(index_number)
    rows: list[tuple[str, str, str, float, int]] = []
    for query_text, query_id, value in queries[
        [query_column, "internal_query_id", query_category_column]
    ].itertuples(index=False, name=None):
        category = "" if pd.isna(value) else str(value)
        indices = (
            category_indices.get(category, []) if category and category != "nan" else []
        )
        tokens = tokenize(str(query_text) if query_text is not None else "")
        ranked = [
            (index_number, index.score(tokens, index.documents[index_number]))
            for index_number in indices
        ]
        ranked = [
            (index_number, score) for index_number, score in ranked if score > 0.0
        ]
        ranked.sort(key=lambda pair: (-pair[1], index.item_ids[pair[0]]))
        if not ranked and category_policy == "fallback":
            ranked = index._retrieve_indices(tokens, min(k, len(index.item_ids)))
        try:
            rows.extend(
                (
                    str(query_id),
                    index.item_ids[index_number],
                    source,
                    float(score),
                    rank,
                )
                for rank, (index_number, score) in enumerate(ranked[:k], 1)
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("invalid category-filtered BM25 scores") from exc
    return pd.DataFrame(
        rows,
        columns=["internal_query_id", "item_id", "source", "score", "rank"],
    )
