"""Tests for TFIDFIndex, retrieve_tfidf and metric evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from avito_candidate_generation.candidates import validate_candidates
from avito_candidate_generation.evaluation import recall_at_k
from avito_candidate_generation.retrievers.tfidf import TFIDFIndex, retrieve_tfidf


def test_tfidf_index_fit_and_retrieve_basic() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["item_car", "item_phone", "item_house"],
            "text": [
                "продажа авто ремонт автомобиля",
                "ремонт телефонов смартфонов iphone",
                "аренда дома коттеджа посуточно",
            ],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "text": ["автомобиль ремонт", "смартфонов iphone"],
        }
    )

    index = TFIDFIndex.fit(items)
    results = index.retrieve(queries, k=2)

    validate_candidates(results)
    assert len(results) == 4  # 2 queries * 2 candidates
    assert set(results.columns) == {
        "internal_query_id",
        "item_id",
        "source",
        "score",
        "rank",
    }

    # Query 1 should retrieve item_car as top candidate
    q1_top = (
        results[results["internal_query_id"] == "q1"]
        .sort_values(["rank"])  # pyright: ignore[reportCallIssue]
        .iloc[0]
    )
    assert q1_top["item_id"] == "item_car"
    assert q1_top["score"] > 0.0
    assert q1_top["rank"] == 1

    # Query 2 should retrieve item_phone as top candidate
    q2_top = (
        results[results["internal_query_id"] == "q2"]
        .sort_values(["rank"])  # pyright: ignore[reportCallIssue]
        .iloc[0]
    )
    assert q2_top["item_id"] == "item_phone"
    assert q2_top["score"] > 0.0
    assert q2_top["rank"] == 1


def test_tfidf_index_save_load_round_trip(tmp_path) -> None:
    items = pd.DataFrame(
        {
            "item_id": ["b", "a", "c"],
            "text": ["быстрый синий автомобиль", "красный телефон", "зеленый дом"],
        }
    )
    queries = pd.DataFrame({"internal_query_id": ["q"], "text": ["телефон"]})

    index = TFIDFIndex.fit(items)
    expected = index.retrieve(queries, k=3)

    path = tmp_path / "tfidf_index"
    index.save(path)
    loaded = TFIDFIndex.load(path)
    actual = loaded.retrieve(queries, k=3)

    pd.testing.assert_frame_equal(actual, expected)


def test_tfidf_index_padding_and_zero_matches() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["item_z", "item_a"],
            "text": ["семантика", "лингвистика"],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q_unknown"],
            "text": ["абсолютно_неизвестные_слова_xyz"],
        }
    )

    index = TFIDFIndex.fit(items)
    results = index.retrieve(queries, k=2)

    validate_candidates(results)
    assert len(results) == 2
    # Padded results should have score 0.0 and deterministic sort
    assert results["rank"].tolist() == [1, 2]
    assert results["item_id"].tolist() == ["item_a", "item_z"]
    assert (results["score"] == 0.0).all()


def test_retrieve_tfidf_category_policy() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["item_1", "item_2"],
            "text": ["услуги сантехника", "услуги электрика"],
            "search_category": ["быт", "ремонт"],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1"],
            "text": ["услуги"],
            "search_category": ["быт"],
        }
    )

    # Test global
    res_none = retrieve_tfidf(queries, items, k=2, category_policy="none")
    validate_candidates(res_none)
    assert len(res_none) == 2

    # Test hard category filter: should only search within "быт"
    res_hard = retrieve_tfidf(queries, items, k=2, category_policy="hard")
    validate_candidates(res_hard)
    assert len(res_hard) == 1
    assert res_hard.iloc[0]["item_id"] == "item_1"

    # Test invalid policy
    with pytest.raises(ValueError, match="category_policy"):
        retrieve_tfidf(queries, items, category_policy="unknown")


def test_tfidf_recall_at_k() -> None:
    items = pd.DataFrame(
        {
            "item_id": [f"item_{i}" for i in range(10)],
            "text": [f"товар номер {i} описание детали" for i in range(10)],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q0", "q1"],
            "text": ["номер 0", "номер 1"],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "internal_query_id": ["q0", "q1"],
            "item_id": ["item_0", "item_1"],
        }
    )

    index = TFIDFIndex.fit(items)
    preds = index.retrieve(queries, k=5)

    r1 = recall_at_k(preds, ground_truth, k=1)
    r5 = recall_at_k(preds, ground_truth, k=5)
    assert r1 == 1.0
    assert r5 == 1.0
