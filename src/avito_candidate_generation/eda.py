"""Machine-readable descriptive diagnostics and CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def describe_table(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        nulls = {str(k): int(v) for k, v in frame.isna().sum().items()}
        duplicates = int(frame.duplicated().sum())
    except (TypeError, ValueError) as exc:
        raise ValueError("unable to describe table") from exc
    return {
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "nulls": nulls,
        "duplicates": duplicates,
    }


def write_eda_report(tables: dict[str, pd.DataFrame], output: str | Path) -> Path:
    try:
        tables_desc = {name: describe_table(frame) for name, frame in tables.items()}
    except Exception as exc:
        raise ValueError("unable to describe tables for eda report") from exc
    report = {
        "schema_version": 1,
        "tables": tables_desc,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("report",))
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--output", default="artifacts/data/eda_report.json")
    args = parser.parse_args()
    if args.command == "report":
        from .config import load_config, resolve_path

        config = load_config(args.config)
        data = config.values["data"]
        if not isinstance(data, dict):
            raise ValueError("missing [data]")
        tables = {
            name: pd.read_parquet(resolve_path(config, data[key]))
            for name, key in (
                ("train", "train"),
                ("benchmark_queries", "benchmark_queries"),
                ("benchmark_items", "benchmark_items"),
            )
        }
        write_eda_report(tables, resolve_path(config, args.output))


if __name__ == "__main__":
    main()
