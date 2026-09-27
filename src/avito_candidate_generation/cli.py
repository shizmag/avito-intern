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
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("answer.csv"))
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", default="configs/smoke.toml")
    smoke.add_argument("--artifact-root", type=Path, default=Path("artifacts/smoke"))
    verify = commands.add_parser("verify")
    verify.add_argument("--root", default=".")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "smoke":
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
                        args.config, artifact_root=args.artifact_root
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
    else:
        result = run_end_to_end(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command == "verify" and result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
