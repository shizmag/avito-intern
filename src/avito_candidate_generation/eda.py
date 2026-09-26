"""Machine-readable descriptive diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def describe_table(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "nulls": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicates": int(frame.duplicated().sum()),
    }


def write_eda_report(tables: dict[str, pd.DataFrame], output: str | Path) -> Path:
    report = {
        "schema_version": 1,
        "tables": {name: describe_table(frame) for name, frame in tables.items()},
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
