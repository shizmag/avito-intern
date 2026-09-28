"""Run benchmark-scale v2 retrieval research with reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .research import (
    CONTEXT_COLUMNS,
    SparseRetrievalIndex,
    build_research_validation,
    context_folds,
    context_frame,
    field_text,
    ground_truth_map,
    numeric_rank,
    oracle_union_metrics,
    ranking_metrics,
    union_rankings,
    write_candidate_frame,
)

SEED = 42
RETRIEVAL_K = 500


def _file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _rank_to_frame(
    rankings: list[list[tuple[str, float]]], query_ids: list[str], source: str
) -> pd.DataFrame:
    return write_candidate_frame(rankings, query_ids, source=source)


def _microcategory_predictions(
    train: pd.DataFrame,
    evaluation_contexts: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    relevant: dict[str, set[str]],
    *,
    seed: int,
    train_fold: int,
    folds: int,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[dict[str, Any], dict[int, list[set[str]]]]:
    """Fit exact-query multi-label posterior on train contexts only."""
    required = set(CONTEXT_COLUMNS) | {"item_microcat_id"}
    if not required.issubset(train.columns):
        return ({"status": "NOT_RUN", "reason": "missing routing columns"}, {})
    work = context_frame(train)
    folds_frame = context_folds(
        work["internal_query_id"].astype(str), seed=seed, folds=folds
    )
    work = cast(
        pd.DataFrame,
        pd.merge(
            work,
            folds_frame,
            on="internal_query_id",
            how="left",
            validate="many_to_one",
        ),
    )
    training = work[work["fold"] == train_fold].copy()
    if training.empty or evaluation_contexts.empty:
        return ({"status": "NOT_RUN", "reason": "empty train/evaluation contexts"}, {})

    counts: dict[str, dict[str, int]] = {}
    for query, category in zip(
        training["search_query"].astype(str),
        training["item_microcat_id"].astype(str),
        strict=True,
    ):
        query_counts = counts.setdefault(query, {})
        query_counts[category] = query_counts.get(category, 0) + 1
    prior: dict[str, int] = {}
    for query_counts in counts.values():
        for category, count in query_counts.items():
            prior[category] = prior.get(category, 0) + count
    prior_order = [
        category
        for category, _ in sorted(prior.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    predictions: list[list[str]] = []
    for query in evaluation_contexts["search_query"].astype(str):
        query_counts = counts.get(query, {})
        order = query_counts or prior
        predictions.append(
            [
                category
                for category, _ in sorted(
                    order.items(), key=lambda pair: (-pair[1], pair[0])
                )
            ]
            if query_counts
            else prior_order
        )

    benchmark_category = dict(
        zip(
            benchmark_items["item_id"].astype(str).tolist(),
            benchmark_items["item_microcat_id"].astype(str).tolist(),
            strict=True,
        )
    )
    items_by_category: dict[str, set[str]] = {}
    for item_id, category in benchmark_category.items():
        items_by_category.setdefault(category, set()).add(item_id)
    eval_ids = evaluation_contexts["internal_query_id"].astype(str).tolist()
    route_sets: dict[int, list[set[str]]] = {}
    top_k_report: dict[str, Any] = {}
    seen_queries = set(counts)
    for top_k in top_ks:
        category_sets: list[set[str]] = []
        corpus_sizes: list[int] = []
        category_hits: list[float] = []
        item_containment: list[float] = []
        for query_id, predicted in zip(eval_ids, predictions, strict=True):
            selected = set(predicted[:top_k])
            routed_items = set().union(
                *(items_by_category.get(category, set()) for category in selected)
            )
            category_sets.append(routed_items)
            corpus_sizes.append(len(routed_items))
            expected_items = relevant.get(query_id, set())
            expected_categories = {
                benchmark_category[item]
                for item in expected_items
                if item in benchmark_category
            }
            category_hits.append(1.0 if expected_categories & selected else 0.0)
            item_containment.append(
                len(expected_items & routed_items) / max(len(expected_items), 1)
            )
        route_sets[top_k] = category_sets
        sorted_sizes = sorted(corpus_sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0
        p95_index = (
            min(len(sorted_sizes) - 1, len(sorted_sizes) * 95 // 100)
            if sorted_sizes
            else 0
        )
        p95_size = sorted_sizes[p95_index] if sorted_sizes else 0
        top_k_report[str(top_k)] = {
            "category_hit_recall": sum(category_hits) / max(len(category_hits), 1),
            "positive_item_containment": sum(item_containment)
            / max(len(item_containment), 1),
            "corpus_size_mean": sum(corpus_sizes) / max(len(corpus_sizes), 1),
            "corpus_size_median": median_size,
            "corpus_size_p95": p95_size,
            "corpus_size_max": max(corpus_sizes, default=0),
            "route_fraction": sum(corpus_sizes)
            / max(len(corpus_sizes) * len(benchmark_items), 1),
        }
    return (
        {
            "status": "PASS",
            "model": "exact search_query posterior",
            "train_contexts": len(
                set(training["internal_query_id"].astype(str).tolist())
            ),
            "evaluation_contexts": len(evaluation_contexts),
            "seen_search_query_fraction": sum(
                1
                for value in evaluation_contexts["search_query"].astype(str).tolist()
                if value in seen_queries
            )
            / max(len(evaluation_contexts), 1),
            "top_k": top_k_report,
        },
        route_sets,
    )


def _historical_intent_rankings(
    train: pd.DataFrame,
    query_frame: pd.DataFrame,
    query_ids: list[str],
    relevant: dict[str, set[str]],
    benchmark_items: pd.DataFrame,
    *,
    k: int,
    batch_size: int,
    seed: int,
    train_fold: int,
    folds: int,
) -> tuple[list[list[tuple[str, float]]], dict[str, Any]]:
    """Retrieve held-out contexts via train-fold nearest historical contexts."""
    work = context_frame(train)
    folds_frame = context_folds(
        work["internal_query_id"].astype(str), seed=seed, folds=folds
    )
    work = cast(
        pd.DataFrame,
        pd.merge(
            work,
            folds_frame,
            on="internal_query_id",
            how="left",
            validate="many_to_one",
        ),
    )
    train_rows = work[work["fold"] == train_fold].copy()
    if train_rows.empty:
        return (
            [[] for _ in query_ids],
            {"status": "NOT_RUN", "reason": "empty train fold"},
        )

    train_rows_frame = cast(pd.DataFrame, train_rows)
    context_columns = [*CONTEXT_COLUMNS, "internal_query_id"]
    train_contexts: pd.DataFrame = pd.DataFrame(
        {column: train_rows_frame[column].tolist() for column in context_columns}
    ).drop_duplicates()
    train_contexts["text"] = field_text(
        cast(pd.DataFrame, train_contexts),
        ("search_query", "search_infm_params_text"),
    )
    train_index = SparseRetrievalIndex.fit(
        train_contexts["internal_query_id"].astype(str).tolist(),
        train_contexts["text"].astype(str).tolist(),
        name="historical_query",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    historical_queries = field_text(
        query_frame, ("search_query", "search_infm_params_text")
    )
    nearest = train_index.top_k(
        historical_queries, k=min(20, len(train_contexts)), batch_size=batch_size
    )
    positive_item_ids = (
        train_rows.groupby("internal_query_id")["item_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    benchmark_ids = set(benchmark_items["item_id"].astype(str).tolist())
    positive_item_ids = {
        historical_id: {item_id for item_id in item_ids if item_id in benchmark_ids}
        for historical_id, item_ids in positive_item_ids.items()
    }
    rankings: list[list[tuple[str, float]]] = []
    exact_seen = 0
    for neighbors in nearest:
        scores: dict[str, float] = {}
        for neighbor_rank, (historical_id, neighbor_score) in enumerate(
            neighbors, start=1
        ):
            historical_items = positive_item_ids.get(historical_id, set())
            if historical_items:
                exact_seen += 1
            try:
                weight = max(float(neighbor_score), 0.0) / neighbor_rank
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("invalid historical neighbor score") from exc
            for item_id in historical_items:
                scores[item_id] = scores.get(item_id, 0.0) + weight
        ranking = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        rankings.append(ranking)
    recovered_from_history = sum(
        1
        for query_id, ranking in zip(query_ids, rankings, strict=True)
        if {item_id for item_id, _ in ranking} & relevant.get(query_id, set())
    )
    return (
        rankings,
        {
            "status": "PASS",
            "train_contexts": len(train_contexts),
            "exact_or_near_contexts_with_positive_prototype": exact_seen,
            "queries_with_any_hit_at_k": recovered_from_history,
        },
    )


def _evaluate_source(
    rankings: list[list[tuple[str, float]]],
    query_ids: list[str],
    relevant: dict[str, set[str]],
    *,
    candidate_k: int,
) -> dict[str, float]:
    return ranking_metrics(
        rankings, query_ids, relevant, ks=(50, 100, 200, 500, candidate_k)
    )


def run_research(
    *,
    train_path: str | Path = "data/train.parquet",
    benchmark_items_path: str | Path = "data/benchmark_items.parquet",
    output_dir: str | Path = "artifacts/research_v2",
    seed: int = SEED,
    folds: int = 5,
    train_fold: int = 0,
    query_limit: int | None = None,
    k: int = RETRIEVAL_K,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run corrected validation and independent sparse retrieval ablations."""
    started = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(train_path)
    benchmark_items = pd.read_parquet(benchmark_items_path)
    validation = build_research_validation(
        train, benchmark_items, seed=seed, train_fold=train_fold, folds=folds
    )
    print(
        f"[research] validation built: {len(validation.validation_contexts):,} contexts, {len(validation.ground_truth):,} GT pairs",
        flush=True,
    )
    for name, frame in {
        "train_contexts": validation.train_contexts,
        "validation_contexts": validation.validation_contexts,
        "ground_truth": validation.ground_truth,
        "context_folds": validation.context_folds,
    }.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    _json(out / "protocol.json", validation.protocol)

    query_frame = validation.validation_contexts.copy()
    if query_limit is not None and 0 < query_limit < len(query_frame):
        query_frame = query_frame.head(query_limit).copy()
    initial_ids = query_frame["internal_query_id"].astype(str).tolist()
    gt = cast(
        pd.DataFrame,
        validation.ground_truth[
            validation.ground_truth["internal_query_id"].isin(initial_ids)
        ].copy(),
    )
    relevant = ground_truth_map(gt)
    query_frame = cast(
        pd.DataFrame,
        query_frame[query_frame["internal_query_id"].isin(list(relevant))].copy(),
    )
    query_ids = query_frame["internal_query_id"].astype(str).tolist()
    relevant = {query_id: relevant[query_id] for query_id in query_ids}

    print(
        f"[research] microcategory routing: {len(query_frame):,} eval contexts",
        flush=True,
    )
    microcat_report, route_sets = _microcategory_predictions(
        train,
        query_frame,
        benchmark_items,
        relevant,
        seed=seed,
        train_fold=train_fold,
        folds=folds,
    )

    item_ids = benchmark_items["item_id"].astype(str).tolist()
    title_text = field_text(benchmark_items, ("item_title_raw",))
    title_params_text = field_text(
        benchmark_items, ("item_title_raw", "item_infm_params_text")
    )
    description_text = field_text(benchmark_items, ("item_description_raw",))
    params_text = field_text(benchmark_items, ("item_infm_params_text",))
    query_only = field_text(query_frame, ("search_query",))
    query_with_params = field_text(
        query_frame, ("search_query", "search_infm_params_text")
    )
    filters = field_text(query_frame, ("search_infm_params_text",))

    sources: dict[str, list[list[tuple[str, float]]]] = {}
    source_metrics: dict[str, dict[str, float]] = {}
    title_index: SparseRetrievalIndex | None = None
    for name, texts, index_queries, analyzer, ngrams in (
        ("query_title_word", title_text, query_only, "word", (1, 2)),
        ("query_title_char", title_text, query_only, "char_wb", (3, 5)),
        (
            "query_title_params_word",
            title_params_text,
            query_with_params,
            "word",
            (1, 2),
        ),
        ("query_description_word", description_text, query_only, "word", (1, 2)),
        ("filters_params_word", params_text, filters, "word", (1, 2)),
    ):
        print(f"[research] lexical branch fit/retrieve: {name}", flush=True)
        index = SparseRetrievalIndex.fit(
            item_ids, texts, name=name, analyzer=analyzer, ngram_range=ngrams, min_df=2
        )
        if name == "query_title_word":
            title_index = index
        ranking = index.top_k(index_queries, k=k, batch_size=batch_size)
        sources[name] = ranking
        source_metrics[name] = _evaluate_source(
            ranking, query_ids, relevant, candidate_k=k
        )
        _rank_to_frame(ranking, query_ids, name).to_parquet(
            out / f"candidates_{name}.parquet", index=False
        )

    if title_index is not None and 5 in route_sets:
        routed = title_index.top_k_routed(
            query_only, route_sets[5], k=k, batch_size=batch_size
        )
        sources["query_title_word_routed_top5"] = routed
        source_metrics["query_title_word_routed_top5"] = _evaluate_source(
            routed, query_ids, relevant, candidate_k=k
        )
        _rank_to_frame(routed, query_ids, "query_title_word_routed_top5").to_parquet(
            out / "candidates_query_title_word_routed_top5.parquet", index=False
        )

    print("[research] historical intent branch", flush=True)
    historical_rankings, historical_report = _historical_intent_rankings(
        train,
        query_frame,
        query_ids,
        relevant,
        benchmark_items,
        k=k,
        batch_size=batch_size,
        seed=seed,
        train_fold=train_fold,
        folds=folds,
    )
    sources["historical_intent"] = historical_rankings
    source_metrics["historical_intent"] = _evaluate_source(
        historical_rankings, query_ids, relevant, candidate_k=k
    )
    _rank_to_frame(historical_rankings, query_ids, "historical_intent").to_parquet(
        out / "candidates_historical_intent.parquet", index=False
    )

    print("[research] numeric branch", flush=True)
    numeric = numeric_rank(
        [f"{a} {b}" for a, b in zip(query_with_params, filters, strict=True)],
        [f"{a} {b}" for a, b in zip(title_text, params_text, strict=True)],
        item_ids,
        k=k,
    )
    sources["numeric"] = numeric
    source_metrics["numeric"] = _evaluate_source(
        numeric, query_ids, relevant, candidate_k=k
    )

    lexical_names = [
        "query_title_word",
        "query_title_char",
        "query_title_params_word",
        "query_description_word",
        "filters_params_word",
        "numeric",
        "historical_intent",
    ]
    print("[research] lexical union", flush=True)
    lexical_union = union_rankings([sources[name] for name in lexical_names], limit=k)
    source_metrics["lexical_union_rrf"] = _evaluate_source(
        lexical_union, query_ids, relevant, candidate_k=k
    )
    source_metrics["oracle_union"] = oracle_union_metrics(
        [sources[name] for name in lexical_names], query_ids, relevant, ks=(500, k)
    )
    _rank_to_frame(lexical_union, query_ids, "lexical_union_rrf").to_parquet(
        out / "candidates_lexical_union_rrf.parquet", index=False
    )

    baseline_path = Path("artifacts/real_validation/dense_e5_base_validation.json")
    dense_baseline: dict[str, Any] = {"status": "NOT_AVAILABLE"}
    if baseline_path.is_file():
        try:
            dense_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid dense baseline artifact") from exc

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "protocol": validation.protocol,
        "counts": {
            "evaluated_queries": len(query_ids),
            "ground_truth_pairs": len(gt),
            "benchmark_items": len(benchmark_items),
        },
        "metrics": source_metrics,
        "microcategory": microcat_report,
        "historical_intent": historical_report,
        "dense_generic_baseline": dense_baseline,
        "timing_seconds": {"total": round(time.time() - started, 3)},
        "config": {
            "seed": seed,
            "folds": folds,
            "train_fold": train_fold,
            "k": k,
            "batch_size": batch_size,
        },
        "fingerprints": {
            "train": _file_sha256(train_path),
            "benchmark_items": _file_sha256(benchmark_items_path),
            "validation_ground_truth": _file_sha256(out / "ground_truth.parquet"),
        },
    }
    _json(out / "metrics.json", report)
    print(f"[research] complete in {time.time() - started:.1f}s", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.parquet")
    parser.add_argument("--benchmark-items", default="data/benchmark_items.parquet")
    parser.add_argument("--output-dir", default="artifacts/research_v2")
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    report = run_research(
        train_path=args.train,
        benchmark_items_path=args.benchmark_items,
        output_dir=args.output_dir,
        query_limit=args.query_limit,
        k=args.k,
        batch_size=args.batch_size,
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
