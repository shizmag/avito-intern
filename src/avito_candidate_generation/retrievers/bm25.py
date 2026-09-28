"""Deterministic BM25 retrieval backed by the maintained :mod:`bm25s` library."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

import bm25s
import numpy as np
import pandas as pd

from ..preprocessing import tokenize

logging.getLogger("bm25s").setLevel(logging.WARNING)


@dataclass
class BM25Index:
    """Thin stable-ID adapter around ``bm25s.BM25``."""

    item_ids: list[str]
    documents: list[tuple[str, ...]]
    k1: float = 1.5
    b: float = 0.75
    _retriever: bm25s.BM25 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.documents):
            raise ValueError("item_ids and documents must have equal length")
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b, method="robertson")
        self._retriever.index(self.documents, show_progress=False)

    @classmethod
    def fit(cls, items: pd.DataFrame, text_column: str = "text") -> BM25Index:
        return cls(
            item_ids=items["item_id"].astype(str).tolist(),
            documents=[tuple(tokenize(value)) for value in items[text_column].fillna("").astype(str)],
        )

    def score(self, query: Sequence[str], document: tuple[str, ...]) -> float:
        """Return bm25s score for one indexed document, retained for compatibility."""
        try:
            document_index = self.documents.index(document)
        except ValueError:
            return 0.0
        indices, scores = self._retriever.retrieve(
            [list(query)],
            k=len(self.item_ids),
            show_progress=False,
        )
        matches = np.flatnonzero(indices[0] == document_index)
        if not matches.size:
            return 0.0
        return scores[0].tolist()[matches.tolist()[0]]

    def _retrieve_one(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self.item_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        count = min(k, len(self.item_ids))
        indices, scores = self._retriever.retrieve(
            [list(tokens)], k=count, show_progress=False
        )
        index_values: list[int] = indices[0].tolist()
        score_values: list[float] = scores[0].tolist()
        results = [
            (self.item_ids[index], score)
            for index, score in zip(index_values, score_values, strict=True)
            if score > 0.0
        ]
        results.sort(key=lambda value: (-value[1], value[0]))
        seen = {item_id for item_id, _ in results}
        if len(results) < count:
            results.extend(
                (item_id, 0.0)
                for item_id in sorted(self.item_ids)
                if item_id not in seen
            )
        return results[:k]

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_column: str = "text",
        k: int = 50,
        source: str = "bm25",
    ) -> pd.DataFrame:
        rows: list[tuple[str, str, str, float, int]] = []
        for query_text, query_id in queries[[query_column, "internal_query_id"]].itertuples(
            index=False, name=None
        ):
            ranked = self._retrieve_one(str(query_text) if query_text is not None else "", k)
            rows.extend(
                (str(query_id), item_id, source, score, rank)
                for rank, (item_id, score) in enumerate(ranked, start=1)
            )
        return pd.DataFrame(
            rows,
            columns=["internal_query_id", "item_id", "source", "score", "rank"],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists() and target.is_file():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        self._retriever.save(target, corpus=None)
        (target / "metadata.json").write_text(
            json.dumps(
                {
                    "item_ids": self.item_ids,
                    "documents": self.documents,
                    "k1": self.k1,
                    "b": self.b,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> BM25Index:
        target = Path(path)
        try:
            if not target.is_dir():
                raise ValueError("BM25 index artifact must be a directory")
            metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            index = BM25Index.__new__(BM25Index)
            index.item_ids = [str(value) for value in metadata["item_ids"]]
            index.documents = [tuple(value) for value in metadata["documents"]]
            index.k1 = float(metadata["k1"])
            index.b = float(metadata["b"])
            if len(index.item_ids) != len(index.documents):
                raise ValueError("invalid BM25 index")
            index._retriever = bm25s.BM25.load(target, load_corpus=False, mmap=False)
            return index
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
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
    """Retrieve globally or from per-category corpora before top-k selection."""
    if category_policy not in {"none", "hard", "fallback"}:
        raise ValueError("category_policy must be one of none, hard, fallback")
    global_index = BM25Index.fit(items, text_column)
    if category_policy == "none":
        return global_index.retrieve(queries, query_column, k, source)
    if query_category_column not in queries or item_category_column not in items:
        raise ValueError("category policy requires category columns on both tables")

    category_indexes = {
        str(category): BM25Index.fit(group, text_column)
        for category, group in items.dropna(subset=[item_category_column]).groupby(
            item_category_column, sort=False
        )
    }
    rows: list[pd.DataFrame] = []
    for category, group in queries.groupby(query_category_column, sort=False, dropna=False):
        category_key = None if category is None or str(category) == "nan" else str(category)
        category_index = category_indexes.get(category_key) if category_key is not None else None
        if category_index is None:
            if category_policy == "hard":
                continue
            category_index = global_index
        candidates = category_index.retrieve(group, query_column, k, source)
        if candidates.empty and category_policy == "fallback":
            candidates = global_index.retrieve(group, query_column, k, source)
        rows.append(candidates)
    if not rows:
        return pd.DataFrame(
            columns=["internal_query_id", "item_id", "source", "score", "rank"]
        )
    return pd.concat(rows, ignore_index=True)
