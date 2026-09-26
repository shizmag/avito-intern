"""Raw and canonical Avito data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class SchemaError(ValueError):
    """Raised when a dataset violates a hard contract."""


ID_LENGTH = 16

TRAIN_REQUIRED = (
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
    "item_id",
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
    "item_category_id",
    "item_location_id",
)
QUERY_REQUIRED = (
    "query_id",
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
)
ITEM_REQUIRED = (
    "item_id",
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
    "item_category_id",
    "item_location_id",
)


@dataclass(frozen=True)
class TableContract:
    name: str
    required_columns: tuple[str, ...]
    id_columns: tuple[str, ...]
    unique_columns: tuple[str, ...]

    def validate(self, frame: pd.DataFrame) -> None:
        require_columns(frame, self.required_columns)
        if frame.empty:
            raise SchemaError(f"{self.name} must not be empty")
        for column in self.id_columns:
            validate_string_ids(frame[column].tolist(), column)
        if self.unique_columns:
            validate_unique(frame, list(self.unique_columns))


def require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise SchemaError(f"missing required columns: {missing}")


def validate_string_ids(values: Iterable[object], name: str) -> None:
    for value in values:
        if value is None or not isinstance(value, str) or len(value) != ID_LENGTH:
            raise SchemaError(
                f"{name} must contain non-null strings of length {ID_LENGTH}; got {value!r}"
            )


def validate_unique(frame: pd.DataFrame, columns: list[str]) -> None:
    if bool(frame.duplicated(columns, keep=False).any()):
        raise SchemaError(
            f"duplicate key {columns}: {int(frame.duplicated(columns, keep=False).sum())} rows"
        )


def validate_domains(
    frame: pd.DataFrame, *, numeric_columns: Iterable[str] = ()
) -> dict[str, int]:
    diagnostics: dict[str, int] = {}
    for column in numeric_columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            diagnostics[f"{column}_invalid"] = int(
                ((frame[column].notna()) & pd.isna(values)).sum()
            )  # pyright: ignore[reportArgumentType]
    if "search_is_delivery_search" in frame:
        bad = (
            ~frame["search_is_delivery_search"].isin([0, 1])
            & frame["search_is_delivery_search"].notna()
        )
        if bool(bad.any()):
            raise SchemaError("search_is_delivery_search must be 0/1")
    return diagnostics


TRAIN_CONTRACT = TableContract("train", TRAIN_REQUIRED, ("item_id",), ())
BENCHMARK_QUERIES_CONTRACT = TableContract(
    "benchmark_queries", QUERY_REQUIRED, ("query_id",), ("query_id",)
)
BENCHMARK_ITEMS_CONTRACT = TableContract(
    "benchmark_items", ITEM_REQUIRED, ("item_id",), ("item_id",)
)
