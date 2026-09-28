"""Leakage-safe CatBoost fusion selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SelectorModel(dict[str, object]):
    """Serialized CatBoost selector metadata and model path."""


def _feature_columns(features: pd.DataFrame, label_column: str) -> list[str]:
    excluded = {label_column, "internal_query_id", "item_id", "fold"}
    columns = [
        str(column)
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not columns:
        raise ValueError("no numeric fusion features")
    return columns


def train_selector(
    features: pd.DataFrame,
    label_column: str = "label",
    *,
    iterations: int = 100,
    depth: int = 6,
    learning_rate: float = 0.05,
    random_seed: int = 42,
    thread_count: int | None = None,
) -> SelectorModel:
    """Fit CatBoostClassifier on a fixed numeric OOF feature schema."""
    if label_column not in features:
        raise ValueError("label column missing")
    if iterations < 1 or depth < 1 or learning_rate <= 0:
        raise ValueError("invalid CatBoost training parameters")
    columns = _feature_columns(features, label_column)
    labels = pd.Series(
        pd.to_numeric(features[label_column], errors="raise"), index=features.index
    ).astype("int8")
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    try:
        from catboost import CatBoostClassifier  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CatBoost fusion requires catboost dependency") from exc
    import os

    threads = (
        thread_count
        if thread_count is not None
        else min(os.cpu_count() or 4, 8)
    )
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=threads,
    )
    model.fit(features[columns], labels)
    return SelectorModel(
        {
            "schema_version": 2,
            "algorithm": "CatBoostClassifier",
            "feature_columns": columns,
            "model": model,
        }
    )


def _validate_model(model: dict[str, object]) -> tuple[list[str], Any]:
    columns = model.get("feature_columns")
    estimator = model.get("model")
    if (
        model.get("schema_version") != 2
        or model.get("algorithm") != "CatBoostClassifier"
        or not isinstance(columns, list)
        or estimator is None
    ):
        raise ValueError("invalid CatBoost selector model")
    return [str(column) for column in columns], estimator


def score_selector(features: pd.DataFrame, model: dict[str, object]) -> pd.Series:
    columns, estimator = _validate_model(model)
    missing = sorted(set(columns).difference(features.columns))
    if missing:
        raise ValueError(f"missing fusion features: {missing}")
    values = estimator.predict_proba(features[columns])[:, 1]
    return pd.Series(values, index=features.index, dtype="float64")


def save_selector(model: dict[str, object], path: str | Path) -> Path:
    columns, estimator = _validate_model(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model_path = target.with_suffix(".cbm")
    estimator.save_model(str(model_path))
    metadata = {
        "schema_version": 2,
        "algorithm": "CatBoostClassifier",
        "feature_columns": columns,
        "model_file": model_path.name,
    }
    target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target


def load_selector(path: str | Path) -> SelectorModel:
    target = Path(path)
    try:
        metadata = json.loads(target.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != 2
            or metadata.get("algorithm") != "CatBoostClassifier"
        ):
            raise ValueError("invalid CatBoost selector artifact")
        columns = metadata["feature_columns"]
        model_file = target.parent / metadata["model_file"]
        from catboost import CatBoostClassifier  # type: ignore[import-not-found]

        estimator = CatBoostClassifier()
        estimator.load_model(str(model_file))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid CatBoost selector artifact") from exc
    return SelectorModel(
        {
            "schema_version": 2,
            "algorithm": "CatBoostClassifier",
            "feature_columns": columns,
            "model": estimator,
        }
    )
