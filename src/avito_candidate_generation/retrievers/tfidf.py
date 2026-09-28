"""Deterministic TF-IDF retrieval backed by scikit-learn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ..candidates import validate_candidates


@dataclass
class TFIDFIndex:
    """Stable-ID TF-IDF index using sparse cosine similarity."""

    item_ids: list[str]
    vectorizer: TfidfVectorizer
    item_matrix: sp.csr_matrix

    def __post_init__(self) -> None:
        n_rows = self.item_matrix.get_shape()[0]
        if n_rows is not None and len(self.item_ids) != n_rows:
            raise ValueError("item_ids and item_matrix rows must have equal length")

    @classmethod
    def fit(
        cls,
        items: pd.DataFrame,
        text_column: str = "text",
        *,
        sublinear_tf: bool = True,
        token_pattern: str = r"[\w]+",
        min_df: int = 1,
        max_features: int | None = None,
    ) -> TFIDFIndex:
        """Fit a TF-IDF vectorizer and construct the item document matrix."""
        item_ids = items["item_id"].astype(str).tolist()
        texts = items[text_column].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=token_pattern,
            sublinear_tf=sublinear_tf,
            norm="l2",
            min_df=min_df,
            max_features=max_features,
        )
        item_matrix = vectorizer.fit_transform(texts)
        if not isinstance(item_matrix, sp.csr_matrix):
            item_matrix = item_matrix.tocsr()
        return cls(item_ids=item_ids, vectorizer=vectorizer, item_matrix=item_matrix)

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_column: str = "text",
        k: int = 50,
        source: str = "tfidf",
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Retrieve top-k items per query by cosine similarity."""
        if k <= 0 or not self.item_ids or queries.empty:
            return pd.DataFrame(
                columns=["internal_query_id", "item_id", "source", "score", "rank"]
            )

        count = min(k, len(self.item_ids))
        all_sorted_ids = sorted(self.item_ids)

        rows: list[tuple[str, str, str, float, int]] = []
        query_texts = queries[query_column].fillna("").astype(str).tolist()
        query_ids = queries["internal_query_id"].astype(str).tolist()

        for start in range(0, len(queries), batch_size):
            b_texts = query_texts[start : start + batch_size]
            b_qids = query_ids[start : start + batch_size]

            q_mat = self.vectorizer.transform(b_texts)
            sims = (q_mat @ self.item_matrix.T).tocsr()

            for row_idx, qid in enumerate(b_qids):
                ptr_start = sims.indptr[row_idx]
                ptr_end = sims.indptr[row_idx + 1]

                col_indices = sims.indices[ptr_start:ptr_end]
                scores = sims.data[ptr_start:ptr_end]

                positive_mask = scores > 0.0
                pos_indices = col_indices[positive_mask]
                pos_scores = scores[positive_mask]

                if len(pos_scores) > 0:
                    if len(pos_scores) > count:
                        top_part = np.argpartition(-pos_scores, count)[:count]
                        top_indices = pos_indices[top_part]
                        top_scores = pos_scores[top_part]
                    else:
                        top_indices = pos_indices
                        top_scores = pos_scores

                    ranked = [
                        (self.item_ids[idx], float(sc))
                        for idx, sc in zip(top_indices, top_scores, strict=True)
                    ]
                    ranked.sort(key=lambda item: (-item[1], item[0]))
                else:
                    ranked = []

                seen = {item_id for item_id, _ in ranked}
                if len(ranked) < count:
                    for item_id in all_sorted_ids:
                        if item_id not in seen:
                            ranked.append((item_id, 0.0))
                            if len(ranked) == count:
                                break

                ranked = ranked[:count]
                for rank, (item_id, score) in enumerate(ranked, start=1):
                    rows.append((qid, item_id, source, score, rank))

        frame = pd.DataFrame(
            rows,
            columns=["internal_query_id", "item_id", "source", "score", "rank"],
        )
        validate_candidates(frame)
        return frame

    def save(self, path: str | Path) -> None:
        """Save vectorizer and matrix artifact to directory."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, target / "vectorizer.joblib")
        sp.save_npz(target / "item_matrix.npz", self.item_matrix)
        metadata = {
            "item_ids": self.item_ids,
            "n_items": len(self.item_ids),
            "n_features": int(self.item_matrix.get_shape()[1]),
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> TFIDFIndex:
        """Load TFIDF index from directory."""
        target = Path(path)
        if not target.is_dir():
            raise ValueError("TFIDF index artifact must be a directory")
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        vectorizer = joblib.load(target / "vectorizer.joblib")
        item_matrix = sp.load_npz(target / "item_matrix.npz")
        item_ids = [str(x) for x in metadata["item_ids"]]
        n_rows = item_matrix.get_shape()[0]
        if n_rows is not None and len(item_ids) != n_rows:
            raise ValueError("mismatched item_ids and item_matrix rows")
        return cls(item_ids=item_ids, vectorizer=vectorizer, item_matrix=item_matrix)


def retrieve_tfidf(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    text_column: str = "text",
    query_column: str = "text",
    k: int = 50,
    source: str = "tfidf",
    category_policy: str = "none",
    query_category_column: str = "search_category",
    item_category_column: str = "search_category",
    sublinear_tf: bool = True,
    token_pattern: str = r"[\w]+",
) -> pd.DataFrame:
    """Retrieve globally or per-category candidates using TF-IDF."""
    if category_policy not in {"none", "hard", "fallback"}:
        raise ValueError("category_policy must be one of none, hard, fallback")

    global_index = TFIDFIndex.fit(
        items,
        text_column=text_column,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern,
    )
    if category_policy == "none":
        return global_index.retrieve(queries, query_column, k, source)

    if query_category_column not in queries or item_category_column not in items:
        raise ValueError("category policy requires category columns on both tables")

    category_indexes = {
        str(category): TFIDFIndex.fit(
            group,
            text_column=text_column,
            sublinear_tf=sublinear_tf,
            token_pattern=token_pattern,
        )
        for category, group in items.dropna(subset=[item_category_column]).groupby(
            item_category_column, sort=False
        )
    }

    rows: list[pd.DataFrame] = []
    for category, group in queries.groupby(
        query_category_column, sort=False, dropna=False
    ):
        category_key = (
            None if category is None or str(category) == "nan" else str(category)
        )
        category_index = (
            category_indexes.get(category_key) if category_key is not None else None
        )
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
