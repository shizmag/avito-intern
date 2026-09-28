"""Thin CLI for selected candidate-generation stages."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from avito_candidate_generation.config import load_config, resolve_path
from avito_candidate_generation.data import (
    build_canonical,
    load_table,
    validate_from_config,
)
from avito_candidate_generation.end_to_end import run_end_to_end
from avito_candidate_generation.pipeline import (
    FinalPipeline,
    build_selected_candidates,
    compose_retrieval_tables,
)
from avito_candidate_generation.splits import write_split_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avito-candidate-generation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/selected.toml")
    prepare.add_argument("--allow-network", action="store_true")
    real_validation = commands.add_parser("real-validation")
    real_validation.add_argument("--config", default="configs/selected.toml")
    real_validation.add_argument(
        "--output", default="artifacts/metrics/real_validation.json"
    )
    real_validation.add_argument("--batch-size", type=int, default=256)
    tfidf_val = commands.add_parser("tfidf-validation")
    tfidf_val.add_argument("--config", default="configs/selected.toml")
    tfidf_val.add_argument("--subset", type=int, default=1000)
    tfidf_val.add_argument(
        "--output", default="artifacts/metrics/tfidf_validation.json"
    )
    tfidf_val.add_argument("--batch-size", type=int, default=256)
    tfidf_val.add_argument("--seed", type=int, default=42)
    rerank_val = commands.add_parser("rerank-validation")
    rerank_val.add_argument("--config", default="configs/selected.toml")
    rerank_val.add_argument("--subset", type=int, default=500)
    rerank_val.add_argument("--candidate-k", type=int, default=500)
    rerank_val.add_argument("--device", default="auto")
    rerank_val.add_argument("--batch-size", type=int, default=64)
    rerank_val.add_argument("--cache-dir", default="artifacts/reranker/cache")
    rerank_val.add_argument(
        "--output", default="artifacts/metrics/reranker_validation.json"
    )
    rerank_val.add_argument("--seed", type=int, default=42)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", default="configs/base.toml")
    canonical = commands.add_parser("canonicalize")
    canonical.add_argument("--config", default="configs/base.toml")
    canonical.add_argument("--output-dir", default="artifacts/data")
    split = commands.add_parser("split")
    split.add_argument("--interactions", required=True, type=Path)
    split.add_argument("--items", required=True, type=Path)
    split.add_argument("--output-dir", default="artifacts/splits")
    split.add_argument("--seed", type=int, default=42)
    candidates = commands.add_parser("candidates")
    candidates.add_argument("--queries", type=Path, required=True)
    candidates.add_argument("--items", type=Path, required=True)
    candidates.add_argument(
        "--output", type=Path, default=Path("artifacts/selected/candidates.parquet")
    )
    candidates.add_argument("--manifest", type=Path, default=Path("artifacts/selected"))
    candidates.add_argument("--k", type=int, default=500)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/selected.toml")
    train.add_argument("--artifact-root", type=Path)
    train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume pipeline from existing artifacts if present (default: True)",
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--manifest", type=Path, default=Path("artifacts/selected/manifest.json")
    )
    predict = commands.add_parser("predict")
    predict.add_argument(
        "--manifest", type=Path, default=Path("artifacts/selected/manifest.json")
    )
    predict.add_argument("--output", type=Path, default=Path("answer.csv"))
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", default="configs/smoke.toml")
    smoke.add_argument("--artifact-root", type=Path, default=Path("artifacts/smoke"))
    verify = commands.add_parser("verify")
    verify.add_argument("--root", default=".")
    verify.add_argument(
        "--scope", choices=("structural", "release"), default="structural"
    )
    verify.add_argument("--submission", type=Path)
    verify.add_argument("--queries", type=Path)
    verify.add_argument("--items", type=Path)
    ans = commands.add_parser(
        "answer", help="Generate final verified competition submission answer.csv"
    )
    ans.add_argument("--output", type=Path, default=Path("answer.csv"))
    ans.add_argument("--routing-k", type=int, default=10)
    ans.add_argument("--dist-decay", type=float, default=0.25)
    ans.add_argument("--same-loc-bonus", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        from avito_candidate_generation.provisioning import prepare

        result = prepare(args.config, allow_network=args.allow_network)
    elif args.command == "real-validation":
        from avito_candidate_generation.validation import run_bm25_validation

        result = run_bm25_validation(
            args.config, output=args.output, batch_size=args.batch_size
        )
    elif args.command == "tfidf-validation":
        from avito_candidate_generation.validation import run_tfidf_validation

        result = run_tfidf_validation(
            args.config,
            output=args.output,
            subset_queries=args.subset,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.command == "rerank-validation":
        from avito_candidate_generation.rerank_experiment import (
            run_reranker_validation,
        )

        result = run_reranker_validation(
            args.config,
            sample_size=args.subset,
            candidate_k=args.candidate_k,
            device=args.device,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            output_json=args.output,
            seed=args.seed,
        )
    elif args.command == "smoke":
        workflow = cast(Any, import_module("avito_candidate_generation.workflow"))
        result = {
            "manifest": str(
                workflow.train_selected(
                    args.config,
                    artifact_root=args.artifact_root,
                    encoder=workflow.SmokeDenseEncoder(),
                )
            )
        }
    elif args.command in {"train", "evaluate", "predict"}:
        workflow = cast(Any, import_module("avito_candidate_generation.workflow"))
        if args.command == "train":
            result = {
                "manifest": str(
                    workflow.train_selected(
                        args.config,
                        artifact_root=args.artifact_root,
                        resume=args.resume,
                    )
                )
            }
        elif args.command == "evaluate":
            result = workflow.evaluate_selected(args.manifest)
        else:
            result = {
                "output": str(
                    workflow.predict_selected(args.manifest, output_csv=args.output)
                )
            }
    elif args.command == "validate":
        result = validate_from_config(args.config)
    elif args.command == "canonicalize":
        config = load_config(args.config)
        data = config.values["data"]
        paths = {key: resolve_path(config, value) for key, value in data.items()}
        result = {
            key: str(value)
            for key, value in build_canonical(
                paths["train"],
                paths["benchmark_queries"],
                paths["benchmark_items"],
                args.output_dir,
            ).items()
        }
    elif args.command == "candidates":
        query_frame = load_table(args.queries)
        item_frame = load_table(args.items)
        query_text, item_text = compose_retrieval_tables(query_frame, item_frame)
        FinalPipeline.load(args.manifest)
        candidates_frame = build_selected_candidates(
            query_text, item_text, retrieval_k=args.k
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        candidates_frame.to_parquet(args.output, index=False)
        result = {
            "output": str(args.output),
            "rows": len(candidates_frame),
            "manifest": str(args.manifest),
        }
    elif args.command == "split":
        interactions = load_table(args.interactions)
        items = load_table(args.items)
        result = {
            key: str(value)
            for key, value in write_split_artifacts(
                interactions,
                items["item_id"].astype(str).tolist(),
                args.output_dir,
                seed=args.seed,
            ).items()
        }
    elif args.command == "answer":
        from avito_candidate_generation.answer_pipeline import generate_champion_answer

        result = generate_champion_answer(
            output_path=args.output,
            routing_k=args.routing_k,
            dist_decay=args.dist_decay,
            same_loc_bonus=args.same_loc_bonus,
        )
    elif args.command == "verify" and args.submission is not None:
        from avito_candidate_generation.submission import validate_submission

        if args.queries is None or args.items is None:
            raise SystemExit("--queries and --items are required with --submission")
        report = validate_submission(args.submission, args.queries, args.items)
        result = {
            "status": report.status,
            "checks": report.checks,
            "counts": report.counts,
        }
    else:
        result = run_end_to_end(args.root, strict=args.scope == "release")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command == "verify" and result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
