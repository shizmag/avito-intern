"""Leakage-safe fusion selector interface with deterministic linear fallback."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def train_selector(
    features: pd.DataFrame, label_column: str = "label"
) -> dict[str, object]:
    if label_column not in features:
        raise ValueError("label column missing")
    excluded = {label_column, "internal_query_id", "item_id", "fold"}
    columns = [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not columns:
        raise ValueError("no numeric fusion features")
    positives = features.loc[features[label_column] == 1, columns].mean()
    negatives = features.loc[features[label_column] == 0, columns].mean()
    try:
        weights = (positives - negatives).fillna(0.0).to_dict()
    except (TypeError, AttributeError) as exc:
        raise ValueError("fusion features must be numeric") from exc
    return {
        "feature_columns": columns,
        "weights": {str(k): float(v) for k, v in weights.items()},
        "schema_version": 1,
    }


def score_selector(features: pd.DataFrame, model: dict[str, object]) -> pd.Series:
    columns = model.get("feature_columns")
    weights = model.get("weights")
    if not isinstance(columns, list) or not isinstance(weights, dict):
        raise ValueError("invalid selector model")
    scores = pd.Series(0.0, index=features.index)
    for column in columns:
        try:
            values = pd.Series(pd.to_numeric(features[column], errors="coerce"), index=features.index).fillna(0.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid fusion feature: {column}") from exc
        scores = scores + values * float(weights.get(column, 0.0))
    return scores


def save_selector(model: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return target


def load_selector(path: str | Path) -> dict[str, object]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid selector artifact") from exc
