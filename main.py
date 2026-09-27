"""Thin repository entrypoint for selected-pipeline verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avito_candidate_generation.verify import verify_repository


def main() -> None:
    parser = argparse.ArgumentParser(prog="avito-candidate-generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_repository(args.root)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
