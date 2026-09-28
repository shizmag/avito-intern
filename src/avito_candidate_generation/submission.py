"""Strict dtype-safe answer.csv exporter and independent round-trip validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .schemas import ID_LENGTH


@dataclass(frozen=True)
class ValidationReport:
    status: str
    checks: dict[str, bool]
    counts: dict[str, int]
    sha256: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _ids(path: Path, column: str) -> list[str]:
    frame = pd.read_parquet(path, columns=[column])
    values = frame[column].tolist()
    if not all(isinstance(value, str) and len(value) == ID_LENGTH for value in values):
        raise ValueError(f"invalid {column} IDs")
    return [str(value) for value in values]


def validate_submission(
    path: Path, queries_path: Path, items_path: Path
) -> ValidationReport:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    checks: dict[str, bool] = {}
    token_counts: list[int] = []
    try:
        text = raw.decode("utf-8")
        checks["utf8"] = True
        if text.startswith("\ufeff"):
            raise ValueError("BOM")
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        query_ids = _ids(queries_path, "query_id")
        item_ids = set(_ids(items_path, "item_id"))
        checks["columns"] = list(frame.columns) == ["query_id", "answer"]
        checks["row_count"] = len(frame) == len(query_ids)
        checks["query_unique"] = (
            bool(frame["query_id"].is_unique) if "query_id" in frame else False
        )
        checks["query_set"] = (
            set(frame["query_id"].tolist()) == set(query_ids)
            if "query_id" in frame
            else False
        )
        checks["query_order"] = (
            frame["query_id"].tolist() == query_ids if "query_id" in frame else False
        )
        valid_answers = True
        unknown = 0
        duplicate_rows = 0
        answers = frame["answer"].tolist() if "answer" in frame else []
        for answer in answers:
            if answer == "":
                token_counts.append(0)
                continue
            if (
                not isinstance(answer, str)
                or answer != answer.strip()
                or re.search(r"\s{2,}|[\t\r\n]", answer)
            ):
                valid_answers = False
            tokens = answer.split(" ")
            token_counts.append(len(tokens))
            if len(tokens) != len(set(tokens)):
                duplicate_rows += 1
            unknown += sum(
                token not in item_ids or len(token) != ID_LENGTH for token in tokens
            )
        checks["answer_format"] = valid_answers
        checks["count_limit"] = all(0 <= count <= 50 for count in token_counts)
        checks["known_items"] = unknown == 0
        checks["no_duplicate_items"] = duplicate_rows == 0
    except Exception:
        checks["parseable"] = False
    passed = bool(checks) and all(checks.values())
    counts = {
        "rows": len(token_counts),
        "empty_answers": sum(count == 0 for count in token_counts),
        "min_candidates": min(token_counts, default=0),
        "max_candidates": max(token_counts, default=0),
    }
    return ValidationReport("PASS" if passed else "FAIL", checks, counts, sha)


def write_submission(
    predictions_path: Path, queries_path: Path, items_path: Path, output: Path
) -> Path:
    predictions = pd.read_parquet(predictions_path)
    query_ids = _ids(queries_path, "query_id")
    item_ids = set(_ids(items_path, "item_id"))
    required = {"query_id", "item_id", "rank"}
    if not required.issubset(predictions.columns):
        raise ValueError("predictions require query_id, item_id, rank")
    rows: list[dict[str, str]] = []
    for query_id in query_ids:
        part = predictions[predictions["query_id"] == query_id]
        part = part.sort_values(["rank", "item_id"], kind="mergesort")  # pyright: ignore[reportCallIssue]
        identifiers = part["item_id"].tolist()
        valid_ids = all(
            isinstance(value, str) and len(value) == ID_LENGTH and value in item_ids
            for value in identifiers
        )
        if (
            not valid_ids
            or len(identifiers) > 50
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("invalid prediction cardinality or item ID")
        rows.append({"query_id": query_id, "answer": " ".join(identifiers)})
    frame = pd.DataFrame(rows, columns=["query_id", "answer"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".csv", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        report = validate_submission(temporary, queries_path, items_path)
        if not report.passed:
            raise ValueError(f"submission validation failed: {report.checks}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--predictions", type=Path, required=True)
    write.add_argument("--queries", type=Path, required=True)
    write.add_argument("--items", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("validate")
    check.add_argument("--submission", type=Path, required=True)
    check.add_argument("--queries", type=Path, required=True)
    check.add_argument("--items", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "write":
        write_submission(args.predictions, args.queries, args.items, args.output)
        return
    report = validate_submission(args.submission, args.queries, args.items)
    print(json.dumps(asdict(report), indent=2))
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
