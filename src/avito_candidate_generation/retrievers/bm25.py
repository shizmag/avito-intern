"""Small dependency-free BM25 retriever for local deterministic baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from typing import Sequence

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
        for document in self.documents:
            for term in set(document):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
        self.average_length = sum(map(len, self.documents)) / max(
            1, len(self.documents)
        )

    @classmethod
    def fit(cls, items: pd.DataFrame, text_column: str = "text") -> "BM25Index":
        return cls(
            items["item_id"].astype(str).tolist(),
            [tokenize(str(x) if pd.notna(x) else "") for x in items[text_column]],
        )

    def score(self, query: Sequence[str], document: tuple[str, ...]) -> float:
        counts = {term: document.count(term) for term in set(document)}
        result = 0.0
        n = len(self.documents)
        for term in query:
            if term not in counts:
                continue
            df = self.document_frequency.get(term, 0)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            tf = counts[term]
            result += (
                idf
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

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_column: str = "text",
        k: int = 50,
        source: str = "bm25",
    ) -> pd.DataFrame:
        rows: list[tuple[str, str, str, float, int]] = []
        for _, query in queries.iterrows():
            qid = str(query["internal_query_id"])
            value = query[query_column]
            tokens = tokenize(str(value) if value is not None else "")
            order = sorted(
                (
                    (self.score(tokens, doc), item_id)
                    for item_id, doc in zip(self.item_ids, self.documents)
                ),
                key=lambda pair: (-pair[0], pair[1]),
            )[:k]
            rows.extend(
                (qid, item_id, source, float(score), rank)
                for rank, (score, item_id) in enumerate(order, 1)
            )
        return pd.DataFrame(
            rows, columns=["internal_query_id", "item_id", "source", "score", "rank"]
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "BM25Index":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def retrieve_bm25(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    text_column: str = "text",
    query_column: str = "text",
    k: int = 50,
    source: str = "bm25",
) -> pd.DataFrame:
    return BM25Index.fit(items, text_column).retrieve(queries, query_column, k, source)
