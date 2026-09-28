"""Field-aware routed lexical retrieval module."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

_NUMBER_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?(?:кг|г|см|мм|м|л|мл|вт|квм|кв|гб|mb|gb|tb|тб|год(?:а|ы)?|лет|ч|час(?:а|ов)?))?(?!\w)",
    re.IGNORECASE,
)

CANONICAL_BRANCH_NAMES: dict[str, str] = {
    "a": "branch_a",
    "branch_a": "branch_a",
    "query_title": "branch_a",
    "title": "branch_a",
    "b": "branch_b",
    "branch_b": "branch_b",
    "query_title_params": "branch_b",
    "title_params": "branch_b",
    "c": "branch_c",
    "branch_c": "branch_c",
    "query_description": "branch_c",
    "description": "branch_c",
    "d": "branch_d",
    "branch_d": "branch_d",
    "filters_params": "branch_d",
    "params": "branch_d",
}


def normalize_lexical_text(value: object) -> str:
    """Normalize lexical text: NFKC, lowercase, ё->е, normalize whitespace, handles nulls."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(text.split())


_UNIT_NORM: dict[str, str] = {
    "гб": "gb",
    "тб": "tb",
    "мб": "mb",
}


def extract_canonical_numbers(text: object) -> set[str]:
    """Extract canonicalized numbers (including common units) as a set of strings."""
    if text is None:
        return set()
    norm = normalize_lexical_text(text)
    if not norm:
        return set()
    results: set[str] = set()
    for token in _NUMBER_RE.findall(norm):
        clean = re.sub(r"\s+", "", token.replace(",", "."))
        for cyr, lat in _UNIT_NORM.items():
            if clean.endswith(cyr):
                clean = clean[: -len(cyr)] + lat
                break
        results.add(clean)
    return results


def numeric_matching_rank(
    queries: Sequence[str],
    item_texts: Sequence[str],
    item_ids: Sequence[str],
    *,
    k: int,
) -> list[list[tuple[str, float]]]:
    """Retrieve by exact canonical number overlap, with deterministic tie-breaking."""
    if len(item_texts) != len(item_ids):
        raise ValueError("item_texts and item_ids must have equal length")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or not queries or not item_texts:
        return [[] for _ in queries]

    inverted: dict[str, list[int]] = {}
    for index, text in enumerate(item_texts):
        for number in extract_canonical_numbers(text):
            inverted.setdefault(number, []).append(index)

    output: list[list[tuple[str, float]]] = []
    for query in queries:
        query_numbers = extract_canonical_numbers(query)
        if not query_numbers:
            output.append([])
            continue

        counts: dict[int, int] = {}
        for number in query_numbers:
            for index in inverted.get(number, []):
                counts[index] = counts.get(index, 0) + 1

        denom = len(query_numbers)
        ranked = sorted(
            ((str(item_ids[index]), count / denom) for index, count in counts.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        output.append(ranked[:k])
    return output


# Backward compatibility alias
numeric_rank = numeric_matching_rank


@dataclass
class SparseBranchIndex:
    """Deterministic sparse cosine index for a single text field/branch."""

    name: str
    item_ids: list[str]
    vectorizer: TfidfVectorizer
    matrix: sp.csr_matrix
    sorted_item_ids: list[str] = field(init=False)
    item_id_to_idx: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        shape = self.matrix.get_shape()
        n_rows = shape[0] if shape is not None else 0
        if len(self.item_ids) != n_rows:
            raise ValueError(
                f"item_ids ({len(self.item_ids)}) and matrix rows ({n_rows}) must match"
            )
        self.sorted_item_ids = sorted(self.item_ids)
        self.item_id_to_idx = {
            item_id: idx for idx, item_id in enumerate(self.item_ids)
        }

    @classmethod
    def fit(
        cls,
        item_ids: Sequence[str],
        texts: Sequence[str],
        *,
        name: str = "sparse",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        token_pattern: str = r"[\w]+",
    ) -> SparseBranchIndex:
        """Fit a TF-IDF vectorizer and construct the sparse document matrix."""
        if len(item_ids) != len(texts):
            raise ValueError("item_ids and texts must have equal length")
        ids_list = [str(x) for x in item_ids]
        norm_texts = [normalize_lexical_text(t) for t in texts]

        vectorizer_kwargs: dict[str, Any] = {
            "analyzer": analyzer,
            "ngram_range": ngram_range,
            "lowercase": False,  # Texts already normalized and lowercased
            "sublinear_tf": sublinear_tf,
            "norm": "l2",
            "min_df": min_df,
            "max_features": max_features,
        }
        if analyzer == "word":
            vectorizer_kwargs["token_pattern"] = token_pattern

        vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        if len(norm_texts) > 0 and any(t for t in norm_texts):
            matrix = vectorizer.fit_transform(norm_texts)
            if not isinstance(matrix, sp.csr_matrix):
                matrix = matrix.tocsr()
        else:
            matrix = sp.csr_matrix((len(ids_list), 0), dtype=np.float64)

        return cls(name=name, item_ids=ids_list, vectorizer=vectorizer, matrix=matrix)

    def retrieve_routed(
        self,
        query_texts: Sequence[str],
        allowed_item_ids: Sequence[set[str]],
        *,
        k: int,
        global_fallback_k: int = 0,
        pad_zeros: bool = True,
    ) -> list[list[tuple[str, float]]]:
        """Fast routed retrieval strictly from allowed_item_ids up to k with optional global fallback."""
        if len(query_texts) != len(allowed_item_ids):
            raise ValueError("query_texts and allowed_item_ids must have equal length")
        if k < 0 or global_fallback_k < 0:
            raise ValueError("k and global_fallback_k must be non-negative")
        if not query_texts or not self.item_ids:
            return [[] for _ in query_texts]

        norm_queries = [normalize_lexical_text(q) for q in query_texts]

        shape = self.matrix.get_shape()
        n_features = shape[1] if shape is not None else 0
        if (
            hasattr(self.vectorizer, "vocabulary_")
            and self.vectorizer.vocabulary_
            and n_features > 0
        ):
            query_matrix = self.vectorizer.transform(norm_queries)
            sims = (query_matrix @ self.matrix.T).tocsr()
        else:
            sims = None

        results: list[list[tuple[str, float]]] = []

        for row_idx, allowed in enumerate(allowed_item_ids):
            valid_allowed = [
                item_id for item_id in allowed if item_id in self.item_id_to_idx
            ]
            target_k = min(k, len(valid_allowed))

            routed_positive: list[tuple[str, float]] = []
            global_positive: list[tuple[str, float]] = []

            if sims is not None:
                ptr_start = sims.indptr[row_idx]
                ptr_end = sims.indptr[row_idx + 1]
                col_indices = sims.indices[ptr_start:ptr_end]
                scores_data = sims.data[ptr_start:ptr_end]

                for col_idx, score_val in zip(col_indices, scores_data, strict=True):
                    sc = float(score_val)
                    if sc > 0.0:
                        item_id = self.item_ids[col_idx]
                        if item_id in allowed:
                            routed_positive.append((item_id, sc))
                        else:
                            global_positive.append((item_id, sc))

            # 1. Process routed candidates strictly from allowed_item_ids
            routed_positive.sort(key=lambda p: (-p[1], p[0]))
            if len(routed_positive) >= target_k:
                routed_candidates = routed_positive[:target_k]
            else:
                routed_candidates = list(routed_positive)
                if pad_zeros and len(routed_candidates) < target_k:
                    seen = {item_id for item_id, _ in routed_candidates}
                    for item_id in sorted(valid_allowed):
                        if item_id not in seen:
                            routed_candidates.append((item_id, 0.0))
                            if len(routed_candidates) == target_k:
                                break

            # 2. Process global fallback candidates if requested
            if global_fallback_k > 0:
                seen_routed = {item_id for item_id, _ in routed_candidates}
                filtered_global_pos = [
                    (item_id, sc)
                    for item_id, sc in global_positive
                    if item_id not in seen_routed
                ]
                filtered_global_pos.sort(key=lambda p: (-p[1], p[0]))

                max_fallback = min(
                    global_fallback_k, len(self.item_ids) - len(seen_routed)
                )
                if len(filtered_global_pos) >= max_fallback:
                    fallback_candidates = filtered_global_pos[:max_fallback]
                else:
                    fallback_candidates = list(filtered_global_pos)
                    if pad_zeros and len(fallback_candidates) < max_fallback:
                        seen_all = seen_routed | {
                            item_id for item_id, _ in fallback_candidates
                        }
                        for item_id in self.sorted_item_ids:
                            if item_id not in seen_all:
                                fallback_candidates.append((item_id, 0.0))
                                if len(fallback_candidates) == max_fallback:
                                    break
                final_candidates = routed_candidates + fallback_candidates
            else:
                final_candidates = routed_candidates

            results.append(final_candidates)

        return results

    def retrieve(
        self,
        query_texts: Sequence[str],
        *,
        k: int,
        pad_zeros: bool = True,
    ) -> list[list[tuple[str, float]]]:
        """Unconstrained global retrieval (all corpus items allowed)."""
        all_set = set(self.item_ids)
        allowed_list = [all_set] * len(query_texts)
        return self.retrieve_routed(
            query_texts,
            allowed_list,
            k=k,
            global_fallback_k=0,
            pad_zeros=pad_zeros,
        )

    def save(self, path: str | Path) -> None:
        """Save branch index artifacts to directory."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, target / "vectorizer.joblib")
        sp.save_npz(target / "matrix.npz", self.matrix)
        shape = self.matrix.get_shape()
        n_features = int(shape[1]) if shape is not None else 0
        metadata = {
            "name": self.name,
            "item_ids": self.item_ids,
            "n_items": len(self.item_ids),
            "n_features": n_features,
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SparseBranchIndex:
        """Load branch index from directory."""
        target = Path(path)
        meta = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        vectorizer = joblib.load(target / "vectorizer.joblib")
        matrix = sp.load_npz(target / "matrix.npz")
        return cls(
            name=meta["name"],
            item_ids=meta["item_ids"],
            vectorizer=vectorizer,
            matrix=matrix,
        )


@dataclass
class FieldAwareSparseIndex:
    """Field-aware sparse TF-IDF retriever with routed candidate generation."""

    item_ids: list[str]
    branches: dict[str, SparseBranchIndex]

    @property
    def branch_a(self) -> SparseBranchIndex | None:
        """Branch A: search_query <-> item_title_raw (word (1, 2))."""
        return self.branches.get("branch_a")

    @property
    def branch_b(self) -> SparseBranchIndex | None:
        """Branch B: search_query + filters <-> item_title_raw + item_infm_params_text."""
        return self.branches.get("branch_b")

    @property
    def branch_c(self) -> SparseBranchIndex | None:
        """Branch C: search_query <-> item_description_raw (word (1, 2))."""
        return self.branches.get("branch_c")

    @property
    def branch_d(self) -> SparseBranchIndex | None:
        """Branch D: search_infm_params_text <-> item_infm_params_text (word (1, 2))."""
        return self.branches.get("branch_d")

    @classmethod
    def fit(
        cls,
        items: pd.DataFrame | None = None,
        *,
        item_ids: Sequence[str] | None = None,
        item_titles: Sequence[str] | None = None,
        item_descriptions: Sequence[str] | None = None,
        item_params: Sequence[str] | None = None,
        branch_b_analyzer: str = "char_wb",
        branch_b_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        token_pattern: str = r"[\w]+",
        branches: Sequence[str] | None = None,
    ) -> FieldAwareSparseIndex:
        """Fit field-aware TF-IDF indices for branches A, B, C, D."""
        if items is not None:
            if "item_id" in items.columns:
                ids = items["item_id"].astype(str).tolist()
            else:
                ids = [str(i) for i in items.index]

            titles = (
                items["item_title_raw"].fillna("").astype(str).tolist()
                if "item_title_raw" in items.columns
                else (
                    items["title"].fillna("").astype(str).tolist()
                    if "title" in items.columns
                    else [""] * len(ids)
                )
            )
            descriptions = (
                items["item_description_raw"].fillna("").astype(str).tolist()
                if "item_description_raw" in items.columns
                else (
                    items["description"].fillna("").astype(str).tolist()
                    if "description" in items.columns
                    else [""] * len(ids)
                )
            )
            params = (
                items["item_infm_params_text"].fillna("").astype(str).tolist()
                if "item_infm_params_text" in items.columns
                else (
                    items["params"].fillna("").astype(str).tolist()
                    if "params" in items.columns
                    else [""] * len(ids)
                )
            )
        else:
            if item_ids is None:
                raise ValueError("either items DataFrame or item_ids must be provided")
            ids = [str(x) for x in item_ids]
            titles = list(item_titles) if item_titles is not None else [""] * len(ids)
            descriptions = (
                list(item_descriptions)
                if item_descriptions is not None
                else [""] * len(ids)
            )
            params = list(item_params) if item_params is not None else [""] * len(ids)

        if not (len(ids) == len(titles) == len(descriptions) == len(params)):
            raise ValueError(
                "item_ids, titles, descriptions, and params must all have equal length"
            )

        requested_branches = (
            {CANONICAL_BRANCH_NAMES.get(b.lower(), b.lower()) for b in branches}
            if branches is not None
            else {"branch_a", "branch_b", "branch_c", "branch_d"}
        )

        built_branches: dict[str, SparseBranchIndex] = {}

        # Branch A: search_query <-> item_title_raw (word (1, 2))
        if "branch_a" in requested_branches:
            built_branches["branch_a"] = SparseBranchIndex.fit(
                ids,
                titles,
                name="branch_a",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                token_pattern=token_pattern,
            )

        # Branch B: search_query + filters <-> item_title_raw + item_infm_params_text
        if "branch_b" in requested_branches:
            title_params = [
                f"{t} {p}".strip() for t, p in zip(titles, params, strict=True)
            ]
            built_branches["branch_b"] = SparseBranchIndex.fit(
                ids,
                title_params,
                name="branch_b",
                analyzer=branch_b_analyzer,
                ngram_range=branch_b_ngram_range,
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                token_pattern=token_pattern,
            )

        # Branch C: search_query <-> item_description_raw (word (1, 2))
        if "branch_c" in requested_branches:
            built_branches["branch_c"] = SparseBranchIndex.fit(
                ids,
                descriptions,
                name="branch_c",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                token_pattern=token_pattern,
            )

        # Branch D: search_infm_params_text <-> item_infm_params_text (word (1, 2))
        if "branch_d" in requested_branches:
            built_branches["branch_d"] = SparseBranchIndex.fit(
                ids,
                params,
                name="branch_d",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                token_pattern=token_pattern,
            )

        return cls(item_ids=ids, branches=built_branches)

    def retrieve_routed(
        self,
        query_texts: Sequence[str],
        allowed_item_ids: Sequence[set[str]],
        *,
        k: int,
        global_fallback_k: int = 0,
        pad_zeros: bool = True,
        branch: str = "branch_a",
    ) -> list[list[tuple[str, float]]]:
        """Retrieve routed candidates using a designated branch index."""
        canonical = CANONICAL_BRANCH_NAMES.get(branch.lower(), branch.lower())
        if canonical not in self.branches:
            raise KeyError(
                f"branch '{branch}' (canonical '{canonical}') not found in index"
            )

        # Branch D rule: only active when query filters non-empty
        if canonical == "branch_d":
            results: list[list[tuple[str, float]]] = []
            branch_index = self.branches[canonical]
            for q_text, allowed in zip(query_texts, allowed_item_ids, strict=True):
                norm_q = normalize_lexical_text(q_text)
                if not norm_q:
                    results.append([])
                else:
                    single_res = branch_index.retrieve_routed(
                        [norm_q],
                        [allowed],
                        k=k,
                        global_fallback_k=global_fallback_k,
                        pad_zeros=pad_zeros,
                    )
                    results.append(single_res[0])
            return results

        return self.branches[canonical].retrieve_routed(
            query_texts,
            allowed_item_ids,
            k=k,
            global_fallback_k=global_fallback_k,
            pad_zeros=pad_zeros,
        )

    def retrieve_multi_branch(
        self,
        queries: pd.DataFrame | Sequence[str],
        allowed_item_ids: Sequence[set[str]],
        *,
        filter_texts: Sequence[str] | None = None,
        k: int = 50,
        global_fallback_k: int = 0,
        pad_zeros: bool = True,
        branches: Sequence[str] | None = None,
    ) -> dict[str, list[list[tuple[str, float]]]]:
        """Retrieve candidates across multiple field-aware branches."""
        if isinstance(queries, pd.DataFrame):
            q_col = (
                "search_query"
                if "search_query" in queries.columns
                else queries.columns[0]
            )
            search_queries = queries[q_col].fillna("").astype(str).tolist()
            if filter_texts is None and "search_infm_params_text" in queries.columns:
                filter_texts = (
                    queries["search_infm_params_text"].fillna("").astype(str).tolist()
                )
        else:
            search_queries = [str(q) for q in queries]

        if filter_texts is None:
            filters_list = [""] * len(search_queries)
        else:
            filters_list = [str(f) for f in filter_texts]

        if len(search_queries) != len(allowed_item_ids) or len(search_queries) != len(
            filters_list
        ):
            raise ValueError(
                "queries, filter_texts, and allowed_item_ids must have equal length"
            )

        active_branches = (
            [CANONICAL_BRANCH_NAMES.get(b.lower(), b.lower()) for b in branches]
            if branches is not None
            else list(self.branches.keys())
        )

        outputs: dict[str, list[list[tuple[str, float]]]] = {}

        for b_name in active_branches:
            if b_name not in self.branches:
                continue

            if b_name == "branch_a":
                # Branch A: search_query <-> item_title_raw
                outputs[b_name] = self.retrieve_routed(
                    search_queries,
                    allowed_item_ids,
                    k=k,
                    global_fallback_k=global_fallback_k,
                    pad_zeros=pad_zeros,
                    branch="branch_a",
                )
            elif b_name == "branch_b":
                # Branch B: search_query + filters <-> item_title_raw + item_infm_params_text
                combined_queries = [
                    f"{q} {f}".strip()
                    for q, f in zip(search_queries, filters_list, strict=True)
                ]
                outputs[b_name] = self.retrieve_routed(
                    combined_queries,
                    allowed_item_ids,
                    k=k,
                    global_fallback_k=global_fallback_k,
                    pad_zeros=pad_zeros,
                    branch="branch_b",
                )
            elif b_name == "branch_c":
                # Branch C: search_query <-> item_description_raw
                outputs[b_name] = self.retrieve_routed(
                    search_queries,
                    allowed_item_ids,
                    k=k,
                    global_fallback_k=global_fallback_k,
                    pad_zeros=pad_zeros,
                    branch="branch_c",
                )
            elif b_name == "branch_d":
                # Branch D: search_infm_params_text <-> item_infm_params_text (only active when filters non-empty)
                outputs[b_name] = self.retrieve_routed(
                    filters_list,
                    allowed_item_ids,
                    k=k,
                    global_fallback_k=global_fallback_k,
                    pad_zeros=pad_zeros,
                    branch="branch_d",
                )

        return outputs

    def save(self, path: str | Path) -> None:
        """Save all field-aware branch index artifacts to directory."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        for b_name, b_index in self.branches.items():
            b_index.save(target / b_name)
        metadata = {
            "item_ids": self.item_ids,
            "branches": list(self.branches.keys()),
        }
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> FieldAwareSparseIndex:
        """Load field-aware sparse index from directory."""
        target = Path(path)
        meta = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        branches: dict[str, SparseBranchIndex] = {}
        for b_name in meta["branches"]:
            branches[b_name] = SparseBranchIndex.load(target / b_name)
        return cls(item_ids=meta["item_ids"], branches=branches)


def retrieve_routed(
    index_or_queries: FieldAwareSparseIndex | SparseBranchIndex | Sequence[str],
    query_texts_or_allowed: Sequence[str] | Sequence[set[str]],
    allowed_item_ids: Sequence[set[str]] | None = None,
    *,
    k: int,
    global_fallback_k: int = 0,
    pad_zeros: bool = True,
    branch: str = "branch_a",
) -> list[list[tuple[str, float]]]:
    """Retrieve routed candidates using index instance or method call dispatch."""
    if isinstance(index_or_queries, FieldAwareSparseIndex):
        if allowed_item_ids is None:
            raise ValueError(
                "allowed_item_ids must be provided when calling retrieve_routed with an index"
            )
        return index_or_queries.retrieve_routed(
            cast(Sequence[str], query_texts_or_allowed),
            allowed_item_ids,
            k=k,
            global_fallback_k=global_fallback_k,
            pad_zeros=pad_zeros,
            branch=branch,
        )
    if isinstance(index_or_queries, SparseBranchIndex):
        if allowed_item_ids is None:
            raise ValueError(
                "allowed_item_ids must be provided when calling retrieve_routed with an index"
            )
        return index_or_queries.retrieve_routed(
            cast(Sequence[str], query_texts_or_allowed),
            allowed_item_ids,
            k=k,
            global_fallback_k=global_fallback_k,
            pad_zeros=pad_zeros,
        )
    raise TypeError(
        "retrieve_routed must be called with a FieldAwareSparseIndex or SparseBranchIndex as first argument"
    )


def rankings_to_dataframe(
    rankings: Sequence[Sequence[tuple[str, float]]],
    query_ids: Sequence[str],
    *,
    source: str = "routed_lexical",
) -> pd.DataFrame:
    """Format candidate rankings into canonical candidate DataFrame."""
    if len(rankings) != len(query_ids):
        raise ValueError("rankings and query_ids must have equal length")
    rows: list[tuple[str, str, str, float, int]] = []
    for qid, query_candidates in zip(query_ids, rankings, strict=True):
        for rank, (item_id, score) in enumerate(query_candidates, start=1):
            rows.append((str(qid), str(item_id), source, float(score), rank))
    return pd.DataFrame(
        rows,
        columns=["internal_query_id", "item_id", "source", "score", "rank"],
    )


__all__ = [
    "FieldAwareSparseIndex",
    "SparseBranchIndex",
    "extract_canonical_numbers",
    "normalize_lexical_text",
    "numeric_matching_rank",
    "numeric_rank",
    "rankings_to_dataframe",
    "retrieve_routed",
]
