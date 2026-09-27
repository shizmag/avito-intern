"""Thin CLI for selected candidate-generation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avito_candidate_generation.config import load_config, resolve_path
from avito_candidate_generation.data import (
    build_canonical,
    load_table,
    validate_from_config,
)
from avito_candidate_generation.end_to_end import run_end_to_end
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
    verify = commands.add_parser("verify")
    verify.add_argument("--root", default=".")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "validate":
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
