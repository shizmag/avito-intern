from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from avito_candidate_generation.workflow import (
    _compose_item_text,
    _compose_query_text,
)


class FixtureDenseEncoder:
    def encode(self, texts, *, batch_size: int) -> np.ndarray:
        return np.asarray(
            [
                [float(len(text)), float(sum(ord(char) for char in text) % 97 + 1)]
                for text in texts
            ],
            dtype=np.float32,
        )


def _write_smoke_inputs(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    items = pd.DataFrame(
        {
            "item_id": ["A" * 16, "B" * 16, "C" * 16],
            "text": ["red phone", "blue car", "green chair"],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "query_id": ["0" * 16, "1" * 16],
            "text": ["red phone", "blue car"],
        }
    )
    pairs = pd.DataFrame(
        {"internal_query_id": ["q1", "q2"], "item_id": ["A" * 16, "B" * 16]}
    )
    for name, frame in [
        ("train_queries", queries.drop(columns="query_id")),
        ("train_items", items),
        ("train_pairs", pairs),
        ("validation_queries", queries),
        ("validation_items", items),
        ("validation_ground_truth", pairs),
        ("benchmark_queries", queries),
        ("benchmark_items", items),
    ]:
        frame.to_parquet(data / f"{name}.parquet", index=False)
    config = root / "selected.toml"
    config.write_text("""[project]
repo_root = "."
seed = 7
[selection]
retrieval_k = 2
[artifacts]
root = "artifacts"
[dense]
model = "fixture"
[two_tower]
dimension = 8
epochs = 1
batch_size = 2
learning_rate = 0.03
[hard_negatives]
max_per_query = 1
[fusion]
rrf_k = 10
[data]
train_queries = "data/train_queries.parquet"
train_items = "data/train_items.parquet"
train_pairs = "data/train_pairs.parquet"
validation_queries = "data/validation_queries.parquet"
validation_items = "data/validation_items.parquet"
validation_ground_truth = "data/validation_ground_truth.parquet"
benchmark_queries = "data/benchmark_queries.parquet"
benchmark_items = "data/benchmark_items.parquet"
""")
    return config


def test_composed_text_preserves_category_metadata() -> None:
    query = pd.DataFrame(
        {
            "internal_query_id": ["q"],
            "search_query": ["phone"],
            "search_category": [123],
        }
    )
    item = pd.DataFrame(
        {"item_id": ["i"], "item_title_raw": ["phone"], "search_category": [123]}
    )
    assert _compose_query_text(query).loc[0, "search_category"] == "123"
    assert _compose_item_text(item).loc[0, "search_category"] == "123"


def test_compose_item_text_preserves_item_category_id() -> None:
    from avito_candidate_generation.workflow import _compose_item_text

    item = pd.DataFrame(
        {"item_id": ["i"], "item_title_raw": ["phone"], "item_category_id": [114]}
    )

    assert _compose_item_text(item).loc[0, "item_category_id"] == "114"


def test_selected_workflow_smoke_is_reproducible(tmp_path: Path) -> None:
    from avito_candidate_generation.workflow import (  # type: ignore[import-not-found]
        evaluate_selected,
        predict_selected,
        train_selected,
    )

    config = _write_smoke_inputs(tmp_path)
    first = train_selected(
        config, artifact_root=tmp_path / "run1", encoder=FixtureDenseEncoder()
    )
    second = train_selected(
        config, artifact_root=tmp_path / "run2", encoder=FixtureDenseEncoder()
    )
    first_manifest = json.loads(first.read_text())
    second_manifest = json.loads(second.read_text())
    assert first_manifest["status"] == "PASS"
    assert first_manifest["validation_metric"] == "recall@50"
    assert (first.parent / "metrics/validation.json").read_text() == (
        second.parent / "metrics/validation.json"
    ).read_text()
    assert evaluate_selected(first)["catboost/recall@50"] >= 0.0
    output = tmp_path / "answer.csv"
    predict_selected(first, output_csv=output)
    assert output.is_file()
    from avito_candidate_generation.submission import validate_submission

    report = validate_submission(
        output,
        Path("tests/fixtures/benchmark_queries.parquet"),
        Path("tests/fixtures/benchmark_items.parquet"),
    )
    assert report.passed
    assert first_manifest["components"] == second_manifest["components"]
