"""Run fast TF-IDF baseline experiment on a small subset of queries/items."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from avito_candidate_generation.config import load_config
from avito_candidate_generation.retrievers.tfidf import TFIDFIndex
from avito_candidate_generation.validation import (
    _atomic_json,
    _compose_item_text,
    _compose_query_text,
    _prepare_workflow_paths,
    _write_metric_csv,
    stream_recall_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast TF-IDF baseline experiment on a subset of the corpus.",
    )
    parser.add_argument(
        "--config",
        default="configs/selected.toml",
        help="Path to workflow config (default: configs/selected.toml)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=1000,
        help="Number of validation queries to sample (default: 1000; 0 for all).",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Optional max item corpus size. Positives are always preserved (default: full validation items).",
    )
    parser.add_argument(
        "--ks",
        default="50,100,150,200,250,300,350,400,450,500",
        help="Comma-separated list of k values (default: 50,100,...,500).",
    )
    parser.add_argument(
        "--sublinear-tf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sublinear tf scaling 1 + log(tf) (default: True).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/metrics/tfidf_validation.json",
        help="Output JSON path (default: artifacts/metrics/tfidf_validation.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Query batch size for retrieval (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    return parser.parse_args()


def sample_subset(
    queries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    items: pd.DataFrame,
    *,
    n_queries: int,
    n_items: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically sample queries and optionally items while preserving positive pairs."""
    from avito_candidate_generation.rerank_experiment import (
        sample_representative_queries,
    )

    # Sample queries
    if 0 < n_queries < len(queries):
        sampled_queries_raw = sample_representative_queries(
            queries, ground_truth, sample_size=n_queries, seed=seed
        )
        sampled_queries = _compose_query_text(sampled_queries_raw)
        valid_qids = list(dict.fromkeys(sampled_queries["internal_query_id"]))
        sampled_gt = pd.DataFrame(
            ground_truth[ground_truth["internal_query_id"].isin(valid_qids)].copy()
        )
    else:
        sampled_queries = _compose_query_text(queries)
        sampled_gt = pd.DataFrame(ground_truth.copy())

    # Preserved item items
    sampled_items = pd.DataFrame(_compose_item_text(items))
    if n_items is not None and 0 < n_items < len(sampled_items):
        pos_item_ids = list(dict.fromkeys(sampled_gt["item_id"].astype(str)))
        pos_items = pd.DataFrame(
            sampled_items[sampled_items["item_id"].isin(pos_item_ids)]
        )
        remaining = pd.DataFrame(
            sampled_items[~sampled_items["item_id"].isin(pos_item_ids)]
        )
        needed = max(0, n_items - len(pos_items))
        if needed > 0 and len(remaining) > needed:
            distractors = remaining.sample(n=needed, random_state=seed)
            sampled_items = pd.DataFrame(
                pd.concat([pos_items, distractors], ignore_index=True)
            )

    return sampled_queries, sampled_gt, sampled_items


def main() -> None:
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    if not ks:
        print("Error: no valid k values provided", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    paths = _prepare_workflow_paths(config, config.values["data"])

    print("Loading validation datasets...")
    raw_queries = pd.read_parquet(paths["validation_queries"])
    raw_gt = pd.read_parquet(paths["validation_ground_truth"])
    raw_items = pd.read_parquet(paths["validation_items"])

    print(
        f"Available validation data: {len(raw_queries):,} queries, "
        f"{len(raw_items):,} items, {len(raw_gt):,} ground-truth pairs."
    )

    queries, ground_truth, items = sample_subset(
        raw_queries,
        raw_gt,
        raw_items,
        n_queries=args.n_queries,
        n_items=args.n_items,
        seed=args.seed,
    )

    n_q = ground_truth["internal_query_id"].nunique()
    n_it = len(items)
    n_rel = len(ground_truth)
    print(
        f"\n[Experiment Setup]\n"
        f"- Sampled queries: {n_q:,}\n"
        f"- Item pool:       {n_it:,}\n"
        f"- Ground-truth:    {n_rel:,} pairs\n"
        f"- Sublinear TF:    {args.sublinear_tf}\n"
        f"- Evaluated Ks:    {ks}\n"
    )

    t0 = time.time()
    print("Fitting TF-IDF index on item texts...")
    index = TFIDFIndex.fit(items, sublinear_tf=args.sublinear_tf)
    fit_time = time.time() - t0
    print(
        f"  ✓ Fitted in {fit_time:.2f}s "
        f"(Vocabulary: {len(index.vectorizer.vocabulary_):,} features)"
    )

    print(f"\nEvaluating Recall@K for K in {ks} (batch size: {args.batch_size})...")
    t1 = time.time()
    metrics = stream_recall_at_k(
        queries,
        ground_truth,
        lambda batch, k: index.retrieve(batch, k=k),
        ks=ks,
        batch_size=args.batch_size,
    )
    eval_time = time.time() - t1
    print(
        f"  ✓ Evaluation finished in {eval_time:.2f}s ({eval_time / n_q * 1000:.2f} ms/query)"
    )

    # Print results table
    print("\n" + "=" * 40)
    print(f"{'Metric':<15} | {'Value':>10}")
    print("-" * 40)
    for k in ks:
        key = f"recall@{k}"
        val = metrics.get(key, 0.0)
        print(f"{key:<15} | {val:>10.6f} ({val * 100:>6.2f}%)")
    print("=" * 40)

    # Save artifacts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method": "tfidf",
        "split": f"cold-item-validation-subset-{n_q}",
        "sublinear_tf": args.sublinear_tf,
        "n_queries": n_q,
        "n_items": n_it,
        "n_relevant": n_rel,
        "candidate_k": max(ks),
        "batch_size": args.batch_size,
        "metrics": metrics,
        "timings": {
            "fit_seconds": round(fit_time, 3),
            "eval_seconds": round(eval_time, 3),
        },
        "config_hash": config.hash,
    }
    _atomic_json(output_path, payload)
    _write_metric_csv(output_path.with_suffix(".csv"), payload)
    print(
        f"\nSaved metrics artifact to:\n- {output_path}\n- {output_path.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
