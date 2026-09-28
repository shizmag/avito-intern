"""Canonical Avito table construction and validation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve_path
from .schemas import (
    BENCHMARK_ITEMS_CONTRACT,
    BENCHMARK_QUERIES_CONTRACT,
    ITEM_REQUIRED,
    QUERY_REQUIRED,
    TRAIN_CONTRACT,
    SchemaError,
    validate_domains,
)

DATA_REPORT_VERSION = "data-contract-v1"


def _scalar(value: object) -> str:
    if value is None or (
        not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))
    ):
        return "<NULL>"
    return str(value)


def stable_internal_query_id(
    row: dict[str, Any], fields: list[str] | None = None
) -> str:
    selected = fields or [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    payload = "\x1f".join(f"{field}={_scalar(row.get(field))}" for field in selected)
    return hashlib.sha256(
        (DATA_REPORT_VERSION + "\x1e" + payload).encode("utf-8")
    ).hexdigest()


def load_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def canonical_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(list(frame.columns), kind="mergesort").astype(object)
    payload = {
        "columns": [str(x) for x in frame.columns],
        "rows": len(frame),
        "data": ordered.to_json(orient="records", date_format="iso"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _deduplicate_items(train: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    work = train.sort_values(
        ["item_id"] + [c for c in sorted(train.columns) if c != "item_id"],
        kind="mergesort",
    )
    before = len(work)
    return work.drop_duplicates("item_id", keep="first"), before - work[
        "item_id"
    ].nunique()


def validate_raw(
    train: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame
) -> dict[str, Any]:
    TRAIN_CONTRACT.validate(train)
    BENCHMARK_QUERIES_CONTRACT.validate(queries)
    BENCHMARK_ITEMS_CONTRACT.validate(items)
    validate_domains(
        train,
        numeric_columns=(
            "item_price",
            "item_rating",
            "item_latitude",
            "item_longitude",
        ),
    )
    validate_domains(queries)
    return {
        "train_rows": len(train),
        "benchmark_query_rows": len(queries),
        "benchmark_item_rows": len(items),
        "train_item_ids": len(set(train["item_id"].astype(str).tolist())),  # pyright: ignore[reportArgumentType]
    }  # pyright: ignore[reportArgumentType]


def build_canonical(
    train_path: str | Path,
    benchmark_queries_path: str | Path,
    benchmark_items_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    train = load_table(train_path)
    queries = load_table(benchmark_queries_path, list(QUERY_REQUIRED))
    items = load_table(benchmark_items_path, list(ITEM_REQUIRED))
    diagnostics = validate_raw(train, queries, items)
    query_fields = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    train = train.copy()
    records: list[dict[str, Any]] = train[query_fields].to_dict(orient="records")  # pyright: ignore[reportCallIssue]
    train["internal_query_id"] = [stable_internal_query_id(row) for row in records]
    raw_interactions = len(train)
    interactions = (
        train[["internal_query_id", "item_id"]].drop_duplicates().assign(label=1)
    )
    item_table, item_dropped = _deduplicate_items(train)
    query_table = train[query_fields + ["internal_query_id"]].drop_duplicates(
        subset=["internal_query_id"], keep="first"
    )  # pyright: ignore[reportCallIssue]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "queries_train": query_table,
        "items_train": item_table,
        "interactions": interactions,
        "benchmark_queries": queries,
        "benchmark_items": items,
    }
    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = out / f"{name}.parquet"
        frame.sort_values(list(frame.columns), kind="mergesort").to_parquet(
            path, index=False
        )
        paths[name] = path
    counts: dict[str, Any] = {
        **diagnostics,
        "raw_interactions": raw_interactions,
        "unique_interactions": len(interactions),
        "interaction_duplicates_dropped": raw_interactions - len(interactions),
        "item_snapshots_dropped": item_dropped,
    }
    report = {
        "schema_version": 1,
        "algorithm": DATA_REPORT_VERSION,
        "counts": counts,
        "fingerprints": {
            name: canonical_fingerprint(frame) for name, frame in outputs.items()
        },
    }
    report_path = out / "data_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["report"] = report_path
    return paths


def validate_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    data = config.values.get("data")
    if not isinstance(data, dict):
        raise SchemaError("missing [data]")
    paths = {
        key: resolve_path(config, value)
        for key, value in data.items()
        if isinstance(value, str)
    }  # pyright: ignore[reportArgumentType]
    required = {"train", "benchmark_queries", "benchmark_items"}
    if not required.issubset(paths):
        raise SchemaError("data paths incomplete")
    return validate_raw(
        load_table(paths["train"]),
        load_table(paths["benchmark_queries"]),
        load_table(paths["benchmark_items"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "build-canonical"))
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--output-dir", default="artifacts/data")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config.values["data"]
    if not isinstance(data, dict):
        raise SchemaError("missing [data]")
    if args.command == "validate":
        print(json.dumps(validate_from_config(args.config)))
    else:
        paths = [
            resolve_path(config, data[k])
            for k in ("train", "benchmark_queries", "benchmark_items")
        ]
        build_canonical(
            paths[0], paths[1], paths[2], resolve_path(config, args.output_dir)
        )


if __name__ == "__main__":
    main()
