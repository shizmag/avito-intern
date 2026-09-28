"""Microcategory routing module for search query intent prediction and corpus pruning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


def _format_word_text(query: str, infm_params: str | None, category: str | None) -> str:
    tokens = [str(query).strip()]
    if infm_params:
        p = str(infm_params).strip()
        if p:
            tokens.append(p)
    if category:
        c = str(category).strip()
        if c:
            tokens.append(c)
    return " ".join(tokens)


def _format_char_text(query: str) -> str:
    return str(query).strip()


class ExactPosteriorPredictor:
    """Predicts microcategory distribution based on exact query occurrences in training data.

    Falls back to the prior distribution over all training microcategories for unseen queries.
    Uses deterministic tie-breaking: count descending, then category_id ascending.
    """

    def __init__(self, *, backfill_with_prior: bool = False) -> None:
        self.backfill_with_prior = backfill_with_prior
        self.counts: dict[str, dict[str, int]] = {}
        self.query_totals: dict[str, int] = {}
        self.prior_counts: dict[str, int] = {}
        self.prior_order: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        queries: Sequence[str] | pd.DataFrame,
        categories: Sequence[str] | None = None,
        *,
        query_col: str = "search_query",
        category_col: str = "item_microcat_id",
    ) -> Self:
        """Fit empirical query -> category distributions and global prior."""
        if isinstance(queries, pd.DataFrame):
            if query_col not in queries.columns:
                raise ValueError(f"query_col {query_col!r} not in dataframe")
            if category_col not in queries.columns:
                raise ValueError(f"category_col {category_col!r} not in dataframe")
            query_list = queries[query_col].astype(str).tolist()
            category_list = queries[category_col].astype(str).tolist()
        else:
            if categories is None:
                raise ValueError(
                    "categories must be provided when queries is a sequence"
                )
            query_list = [str(q) for q in queries]
            category_list = [str(c) for c in categories]

        if not query_list or not category_list:
            raise ValueError("training data must not be empty")
        if len(query_list) != len(category_list):
            raise ValueError(
                f"queries ({len(query_list)}) and categories ({len(category_list)}) length mismatch"
            )

        counts: dict[str, dict[str, int]] = {}
        query_totals: dict[str, int] = {}
        prior_counts: dict[str, int] = {}

        for q, c in zip(query_list, category_list, strict=True):
            qc = counts.setdefault(q, {})
            qc[c] = qc.get(c, 0) + 1
            query_totals[q] = query_totals.get(q, 0) + 1
            prior_counts[c] = prior_counts.get(c, 0) + 1

        self.counts = counts
        self.query_totals = query_totals
        self.prior_counts = prior_counts
        self.prior_order = [
            cat
            for cat, _ in sorted(
                prior_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        self._is_fitted = True
        return self

    def support(self, query: str) -> int:
        """Return the number of times the exact query was observed in training."""
        return self.query_totals.get(str(query), 0)

    def is_seen(self, query: str) -> bool:
        """Return whether the exact query was observed in training."""
        return str(query) in self.counts

    def predict_top_k(
        self,
        queries: Sequence[str] | pd.DataFrame,
        k: int,
        *,
        query_col: str = "search_query",
    ) -> list[list[str]]:
        """Predict top-k microcategories for each query."""
        if not self._is_fitted:
            raise ValueError("ExactPosteriorPredictor is not fitted")
        if k < 1:
            raise ValueError("k must be positive")

        if isinstance(queries, pd.DataFrame):
            query_list = queries[query_col].astype(str).tolist()
        else:
            query_list = [str(q) for q in queries]

        results: list[list[str]] = []
        for q in query_list:
            if q in self.counts:
                q_counts = self.counts[q]
                order = [
                    cat
                    for cat, _ in sorted(
                        q_counts.items(), key=lambda pair: (-pair[1], pair[0])
                    )
                ]
                if self.backfill_with_prior and len(order) < k:
                    seen = set(order)
                    order = order + [c for c in self.prior_order if c not in seen]
                results.append(order[:k])
            else:
                results.append(self.prior_order[:k])
        return results

    def predict_proba(
        self,
        queries: Sequence[str] | pd.DataFrame,
        top_k: int,
        *,
        query_col: str = "search_query",
    ) -> list[list[tuple[str, float]]]:
        """Predict top_k microcategories with empirical probabilities for each query."""
        if not self._is_fitted:
            raise ValueError("ExactPosteriorPredictor is not fitted")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if isinstance(queries, pd.DataFrame):
            query_list = queries[query_col].astype(str).tolist()
        else:
            query_list = [str(q) for q in queries]

        total_prior = sum(self.prior_counts.values())
        results: list[list[tuple[str, float]]] = []

        for q in query_list:
            if q in self.counts:
                q_counts = self.counts[q]
                total = self.query_totals[q]
                sorted_items = sorted(
                    q_counts.items(), key=lambda pair: (-pair[1], pair[0])
                )
                items = [(cat, count / total) for cat, count in sorted_items]
                results.append(items[:top_k])
            else:
                sorted_prior = sorted(
                    self.prior_counts.items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )
                items = (
                    [(cat, count / total_prior) for cat, count in sorted_prior]
                    if total_prior > 0
                    else []
                )
                results.append(items[:top_k])
        return results


class GeneralizingClassifierPredictor:
    """Predicts microcategory for queries using TF-IDF features and a linear classifier.

    Features:
    - Word n-grams (1, 2) on query + infm_params_text + category
    - Char_wb n-grams (3, 5) on query alone

    Model:
    - Linear classifier (SGDClassifier with log_loss or LogisticRegression)
    - Deterministic tie-breaking by score/proba desc, category_id asc
    """

    def __init__(
        self,
        classifier: Any | None = None,
        *,
        seed: int = 42,
        alpha: float = 1e-5,
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        self.seed = seed
        self.alpha = alpha
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.classifier = (
            classifier
            if classifier is not None
            else SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                random_state=seed,
            )
        )
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        self.classes_: list[str] = []
        self.single_class_: str | None = None
        self._is_fitted: bool = False

    def _extract_inputs(
        self,
        queries: Sequence[str] | pd.DataFrame,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
        query_col: str = "search_query",
        params_col: str = "search_infm_params_text",
        category_col: str = "search_category",
    ) -> tuple[list[str], list[str | None], list[str | None]]:
        if isinstance(queries, pd.DataFrame):
            col_q = (
                query_col
                if query_col in queries.columns
                else ("query" if "query" in queries.columns else None)
            )
            if col_q is None:
                raise ValueError(
                    f"Neither {query_col!r} nor 'query' column found in DataFrame"
                )
            query_list = queries[col_q].astype(str).tolist()
            col_p = (
                params_col
                if params_col in queries.columns
                else (
                    "infm_params_text"
                    if "infm_params_text" in queries.columns
                    else None
                )
            )
            params_list: list[str | None] = (
                queries[col_p].fillna("").astype(str).tolist()
                if col_p
                else [None] * len(query_list)
            )
            col_c = (
                category_col
                if category_col in queries.columns
                else ("category" if "category" in queries.columns else None)
            )
            cat_list: list[str | None] = (
                queries[col_c].fillna("").astype(str).tolist()
                if col_c
                else [None] * len(query_list)
            )
            return query_list, params_list, cat_list

        query_list = [str(q) for q in queries]
        params_list = (
            [str(p) if p is not None else None for p in infm_params]
            if infm_params is not None
            else [None] * len(query_list)
        )
        cat_list = (
            [str(c) if c is not None else None for c in search_categories]
            if search_categories is not None
            else [None] * len(query_list)
        )
        return query_list, params_list, cat_list

    def fit(
        self,
        queries: Sequence[str] | pd.DataFrame,
        categories: Sequence[str] | None = None,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
        item_microcat_col: str = "item_microcat_id",
    ) -> Self:
        """Fit vectorizers and classifier."""
        query_list, params_list, cat_list = self._extract_inputs(
            queries, infm_params, search_categories
        )
        if isinstance(queries, pd.DataFrame) and categories is None:
            if item_microcat_col not in queries.columns:
                raise ValueError(f"{item_microcat_col!r} not found in DataFrame")
            target_list = queries[item_microcat_col].astype(str).tolist()
        else:
            if categories is None:
                raise ValueError(
                    "categories must be provided when queries is a sequence"
                )
            target_list = [str(c) for c in categories]

        if not query_list or not target_list:
            raise ValueError("training data must not be empty")
        if len(query_list) != len(target_list):
            raise ValueError(
                f"queries ({len(query_list)}) and categories ({len(target_list)}) length mismatch"
            )

        word_texts = [
            _format_word_text(q, p, c)
            for q, p, c in zip(query_list, params_list, cat_list, strict=True)
        ]
        char_texts = [_format_char_text(q) for q in query_list]

        x_word = self.word_vectorizer.fit_transform(word_texts)
        x_char = self.char_vectorizer.fit_transform(char_texts)
        x_feats = hstack([x_word, x_char], format="csr")

        unique_targets = sorted(set(target_list))
        if len(unique_targets) == 1:
            self.single_class_ = unique_targets[0]
            self.classes_ = [unique_targets[0]]
        else:
            self.single_class_ = None
            self.classifier.fit(x_feats, target_list)
            classes = getattr(self.classifier, "classes_", [])
            self.classes_ = [str(c) for c in classes]

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        queries: Sequence[str] | pd.DataFrame,
        top_k: int,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Predict top_k microcategories with calibrated probabilities for each query."""
        if not self._is_fitted:
            raise ValueError("GeneralizingClassifierPredictor is not fitted")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_list, params_list, cat_list = self._extract_inputs(
            queries, infm_params, search_categories
        )
        if not query_list:
            return []

        if self.single_class_ is not None:
            return [[(self.single_class_, 1.0)] for _ in query_list]

        word_texts = [
            _format_word_text(q, p, c)
            for q, p, c in zip(query_list, params_list, cat_list, strict=True)
        ]
        char_texts = [_format_char_text(q) for q in query_list]

        x_word = self.word_vectorizer.transform(word_texts)
        x_char = self.char_vectorizer.transform(char_texts)
        x_feats = hstack([x_word, x_char], format="csr")

        proba_matrix = self.classifier.predict_proba(x_feats)
        results: list[list[tuple[str, float]]] = []

        for row in proba_matrix:
            pairs = [
                (cls_name, float(prob))
                for cls_name, prob in zip(self.classes_, row.tolist(), strict=True)
            ]
            sorted_pairs = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
            results.append(sorted_pairs[:top_k])

        return results

    def predict_top_k(
        self,
        queries: Sequence[str] | pd.DataFrame,
        k: int,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
    ) -> list[list[str]]:
        """Predict top-k microcategories for each query."""
        proba_list = self.predict_proba(
            queries,
            top_k=k,
            infm_params=infm_params,
            search_categories=search_categories,
        )
        return [[cat for cat, _ in top_pairs] for top_pairs in proba_list]


class HybridMicrocategoryPredictor:
    """Combines exact posterior predictor with generalizing classifier.

    - If query is seen in training data with sufficient support (>= min_count),
      uses posterior distribution.
    - Otherwise, blends posterior with generalizing model or falls back to
      generalizing model.
    """

    def __init__(
        self,
        exact_predictor: ExactPosteriorPredictor | None = None,
        generalizing_predictor: GeneralizingClassifierPredictor | None = None,
        *,
        min_count: int = 1,
        blend_strategy: str = "fallback",  # 'fallback' or 'blend'
        blend_alpha: float | None = None,
        seed: int = 42,
    ) -> None:
        self.min_count = min_count
        self.blend_strategy = blend_strategy
        self.blend_alpha = blend_alpha
        self.seed = seed
        self.exact_predictor = (
            exact_predictor
            if exact_predictor is not None
            else ExactPosteriorPredictor()
        )
        self.generalizing_predictor = (
            generalizing_predictor
            if generalizing_predictor is not None
            else GeneralizingClassifierPredictor(seed=seed)
        )
        self._is_fitted: bool = False

    def fit(
        self,
        queries: Sequence[str] | pd.DataFrame,
        categories: Sequence[str] | None = None,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
        item_microcat_col: str = "item_microcat_id",
    ) -> Self:
        """Fit both exact posterior and generalizing classifier."""
        self.exact_predictor.fit(queries, categories, category_col=item_microcat_col)
        self.generalizing_predictor.fit(
            queries,
            categories,
            infm_params=infm_params,
            search_categories=search_categories,
            item_microcat_col=item_microcat_col,
        )
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        queries: Sequence[str] | pd.DataFrame,
        top_k: int,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Predict top_k microcategories with probabilities."""
        if not self._is_fitted:
            if (
                self.exact_predictor._is_fitted
                and self.generalizing_predictor._is_fitted
            ):
                self._is_fitted = True
            else:
                raise ValueError("HybridMicrocategoryPredictor is not fitted")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if isinstance(queries, pd.DataFrame):
            col_q = "search_query" if "search_query" in queries.columns else "query"
            query_list = queries[col_q].astype(str).tolist()
        else:
            query_list = [str(q) for q in queries]

        if not query_list:
            return []

        needs_generalizing = [False] * len(query_list)
        for i, q in enumerate(query_list):
            count = self.exact_predictor.support(q)
            if count >= self.min_count:
                needs_generalizing[i] = False
            else:
                needs_generalizing[i] = True

        gen_probas: list[list[tuple[str, float]] | None] = [None] * len(query_list)
        if any(needs_generalizing):
            gen_indices = [i for i, needed in enumerate(needs_generalizing) if needed]
            sub_queries = (
                queries.iloc[gen_indices]
                if isinstance(queries, pd.DataFrame)
                else [query_list[i] for i in gen_indices]
            )
            sub_params = (
                [infm_params[i] for i in gen_indices]
                if infm_params is not None
                else None
            )
            sub_cats = (
                [search_categories[i] for i in gen_indices]
                if search_categories is not None
                else None
            )
            sub_gen_preds = self.generalizing_predictor.predict_proba(
                sub_queries,
                top_k=max(top_k, len(self.generalizing_predictor.classes_)),
                infm_params=sub_params,
                search_categories=sub_cats,
            )
            for idx, pred in zip(gen_indices, sub_gen_preds, strict=True):
                gen_probas[idx] = pred

        results: list[list[tuple[str, float]]] = []
        for i, q in enumerate(query_list):
            count = self.exact_predictor.support(q)
            if count >= self.min_count:
                exact_p = self.exact_predictor.predict_proba([q], top_k=top_k)[0]
                results.append(exact_p)
            elif count > 0 and self.blend_strategy == "blend":
                exact_p = self.exact_predictor.predict_proba(
                    [q], top_k=len(self.exact_predictor.prior_order)
                )[0]
                gen_p = gen_probas[i] or []
                alpha = (
                    self.blend_alpha
                    if self.blend_alpha is not None
                    else (count / self.min_count)
                )
                blended_scores: dict[str, float] = {}
                for cat, prob in exact_p:
                    blended_scores[cat] = blended_scores.get(cat, 0.0) + alpha * prob
                for cat, prob in gen_p:
                    blended_scores[cat] = (
                        blended_scores.get(cat, 0.0) + (1.0 - alpha) * prob
                    )
                sorted_items = sorted(
                    blended_scores.items(), key=lambda pair: (-pair[1], pair[0])
                )
                results.append(sorted_items[:top_k])
            else:
                gen_p = gen_probas[i]
                if gen_p is not None and gen_p:
                    results.append(gen_p[:top_k])
                else:
                    results.append(
                        self.exact_predictor.predict_proba([q], top_k=top_k)[0]
                    )

        return results

    def predict_top_k(
        self,
        queries: Sequence[str] | pd.DataFrame,
        k: int,
        *,
        infm_params: Sequence[str] | None = None,
        search_categories: Sequence[str] | None = None,
    ) -> list[list[str]]:
        """Predict top-k microcategories for each query."""
        proba_list = self.predict_proba(
            queries,
            top_k=k,
            infm_params=infm_params,
            search_categories=search_categories,
        )
        return [[cat for cat, _ in top_pairs] for top_pairs in proba_list]


def _aggregate_routing_metrics(
    category_hits: list[float],
    item_containments: list[float],
    corpus_sizes: list[int],
    benchmark_corpus_size: int,
) -> dict[str, float | int]:
    """Compute summary metrics and percentiles for a subset of queries."""
    n = len(category_hits)
    if n == 0:
        return {
            "n_queries": 0,
            "category_hit_recall": 0.0,
            "positive_item_containment": 0.0,
            "corpus_size_mean": 0.0,
            "corpus_size_median": 0,
            "corpus_size_p95": 0,
            "corpus_size_max": 0,
            "route_fraction": 0.0,
        }

    sorted_sizes = sorted(corpus_sizes)
    median_size = sorted_sizes[len(sorted_sizes) // 2]
    p95_index = min(len(sorted_sizes) - 1, len(sorted_sizes) * 95 // 100)
    p95_size = sorted_sizes[p95_index]

    return {
        "n_queries": n,
        "category_hit_recall": sum(category_hits) / n,
        "positive_item_containment": sum(item_containments) / n,
        "corpus_size_mean": sum(corpus_sizes) / n,
        "corpus_size_median": median_size,
        "corpus_size_p95": p95_size,
        "corpus_size_max": max(corpus_sizes, default=0),
        "route_fraction": (
            sum(corpus_sizes) / (n * benchmark_corpus_size)
            if benchmark_corpus_size > 0
            else 0.0
        ),
    }


def evaluate_microcategory_routing(
    predictions: dict[int, list[list[str]]],
    query_ids: list[str],
    query_texts: list[str],
    relevant_items: dict[str, set[str]],
    item_to_microcat: dict[str, str],
    seen_queries: set[str],
    benchmark_corpus_size: int,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[Any, Any]:
    """Evaluate candidate routing against ground-truth items and microcategories.

    For each K in top_ks, computes:
    - category_hit_recall: query-mean hit rate on relevant categories
    - positive_item_containment: query-mean fraction of relevant items contained
    - corpus_size_mean, median, p95, max, route_fraction
    - breakdown by: overall, seen_query, unseen_query
    """
    n_queries = len(query_ids)
    if len(query_texts) != n_queries:
        raise ValueError(
            f"query_ids ({n_queries}) and query_texts ({len(query_texts)}) length mismatch"
        )

    # Pre-index items by microcategory
    items_by_cat: dict[str, set[str]] = {}
    for item_id, cat in item_to_microcat.items():
        items_by_cat.setdefault(str(cat), set()).add(str(item_id))

    # Pre-determine seen masks
    is_seen = [str(q) in seen_queries for q in query_texts]

    report: dict[Any, Any] = {
        "status": "PASS",
        "top_ks": top_ks,
        "n_queries": n_queries,
        "seen_queries_count": sum(is_seen),
        "unseen_queries_count": n_queries - sum(is_seen),
        "benchmark_corpus_size": benchmark_corpus_size,
    }

    for k in top_ks:
        if k in predictions:
            pred_k = predictions[k]
        elif predictions:
            max_k = max(predictions.keys())
            pred_k = [p[:k] for p in predictions[max_k]]
        else:
            raise ValueError(f"predictions missing for top_k={k}")

        if len(pred_k) != n_queries:
            raise ValueError(
                f"predictions for k={k} has length {len(pred_k)}, expected {n_queries}"
            )

        cat_hits_overall: list[float] = []
        item_cont_overall: list[float] = []
        sizes_overall: list[int] = []

        cat_hits_seen: list[float] = []
        item_cont_seen: list[float] = []
        sizes_seen: list[int] = []

        cat_hits_unseen: list[float] = []
        item_cont_unseen: list[float] = []
        sizes_unseen: list[int] = []

        for i, qid in enumerate(query_ids):
            selected = set(pred_k[i][:k])
            routed_items = set().union(*(items_by_cat.get(c, set()) for c in selected))
            corpus_size = len(routed_items)
            sizes_overall.append(corpus_size)

            expected_items = relevant_items.get(qid, set())
            expected_categories = {
                item_to_microcat[item]
                for item in expected_items
                if item in item_to_microcat
            }

            cat_hit = (
                1.0
                if (expected_categories and (expected_categories & selected))
                else 0.0
            )
            cat_hits_overall.append(cat_hit)

            item_cont = (
                len(expected_items & routed_items) / len(expected_items)
                if expected_items
                else 0.0
            )
            item_cont_overall.append(item_cont)

            if is_seen[i]:
                cat_hits_seen.append(cat_hit)
                item_cont_seen.append(item_cont)
                sizes_seen.append(corpus_size)
            else:
                cat_hits_unseen.append(cat_hit)
                item_cont_unseen.append(item_cont)
                sizes_unseen.append(corpus_size)

        overall = _aggregate_routing_metrics(
            cat_hits_overall,
            item_cont_overall,
            sizes_overall,
            benchmark_corpus_size,
        )
        seen = _aggregate_routing_metrics(
            cat_hits_seen,
            item_cont_seen,
            sizes_seen,
            benchmark_corpus_size,
        )
        unseen = _aggregate_routing_metrics(
            cat_hits_unseen,
            item_cont_unseen,
            sizes_unseen,
            benchmark_corpus_size,
        )

        k_entry = {
            **overall,
            "overall": overall,
            "seen_query": seen,
            "unseen_query": unseen,
        }
        report[k] = k_entry
        report[str(k)] = k_entry

    return report
