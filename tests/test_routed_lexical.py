"""Unit tests for field-aware routed lexical retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avito_candidate_generation.candidates import validate_candidates
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
    extract_canonical_numbers,
    normalize_lexical_text,
    numeric_matching_rank,
    rankings_to_dataframe,
    retrieve_routed,
)


def test_normalize_lexical_text() -> None:
    # NFKC, lowercase, ё -> е
    assert normalize_lexical_text("  Привет,  Ёлка!\t\n") == "привет, елка!"
    assert normalize_lexical_text("МоЁ СЕМЯ") == "мое семя"

    # Whitespace normalization
    assert normalize_lexical_text("слово1   слово2\nслово3") == "слово1 слово2 слово3"

    # Nulls and special types
    assert normalize_lexical_text(None) == ""
    assert normalize_lexical_text(np.nan) == ""
    assert normalize_lexical_text(float("nan")) == ""
    assert normalize_lexical_text(pd.NA) == ""
    assert normalize_lexical_text("") == ""

    # Non-strings
    assert normalize_lexical_text(12345) == "12345"


def test_extract_canonical_numbers() -> None:
    # Base match requirement and consistency with research test
    assert extract_canonical_numbers("Год 2,0 и 10 кг") == {"2.0", "10кг"}

    # Units handling (with alias canonicalization)
    assert extract_canonical_numbers("iphone 12 128 гб") == {"12", "128gb"}
    assert extract_canonical_numbers("samsung galaxy 256gb") == {"256gb"}
    assert extract_canonical_numbers("диски 17 радиус 5 болтов 112") == {
        "17",
        "5",
        "112",
    }
    assert extract_canonical_numbers("квартира 45,5 кв.м 2 комн") == {"45.5кв", "2"}

    # Nulls
    assert extract_canonical_numbers(None) == set()
    assert extract_canonical_numbers(np.nan) == set()
    assert extract_canonical_numbers("") == set()


def test_numeric_matching_rank() -> None:
    queries = ["iphone 12 128 гб", "bmw x5 2018", "простой запрос без цифр"]
    item_texts = [
        "apple iphone 12 128gb черный",  # matches 12, 128gb (2/2 = 1.0)
        "apple iphone 12 64gb белый",  # matches 12 (1/2 = 0.5)
        "bmw x5 2018 года 3.0",  # matches 2018 (or 2018года)
        "чехол для телефона",  # matches 0
    ]
    item_ids = ["item_iphone_128", "item_iphone_64", "item_bmw", "item_case"]

    rankings = numeric_matching_rank(queries, item_texts, item_ids, k=3)
    assert len(rankings) == 3

    # Query 0: iphone 12 128 гб
    q0_rank = rankings[0]
    assert len(q0_rank) >= 2
    top_item_id, top_score = q0_rank[0]
    assert top_item_id == "item_iphone_128"
    assert top_score == 1.0

    second_item_id, second_score = q0_rank[1]
    assert second_item_id == "item_iphone_64"
    assert second_score == 0.5

    # Query 2 has no numbers -> returns empty candidate list
    assert rankings[2] == []

    # Deterministic tie-breaking for equal overlap score
    q_tie = ["диски 17"]
    items_tie = ["диски 17 диаметр", "колеса 17 дюймов"]
    ids_tie = ["item_z", "item_a"]
    tie_rank = numeric_matching_rank(q_tie, items_tie, ids_tie, k=2)[0]
    # Both have score 1.0 -> tie broken by item_id ascending ("item_a" before "item_z")
    assert tie_rank[0][0] == "item_a"
    assert tie_rank[1][0] == "item_z"


def test_sparse_branch_deterministic_retrieval() -> None:
    item_ids = ["item_car", "item_phone", "item_laptop"]
    texts = [
        "продажа автомобиля lada vesta",
        "смартфон apple iphone 14 pro",
        "ноутбук apple macbook pro m2",
    ]
    index = SparseBranchIndex.fit(item_ids, texts, name="test_branch")

    query = ["apple iphone"]
    allowed = [{"item_car", "item_phone", "item_laptop"}]

    # Run retrieval twice to ensure 100% determinism
    res1 = index.retrieve_routed(query, allowed, k=2)
    res2 = index.retrieve_routed(query, allowed, k=2)

    assert res1 == res2
    assert len(res1[0]) == 2
    assert res1[0][0][0] == "item_phone"
    assert res1[0][0][1] > 0.0


def test_routed_filtering_strictly_enforces_allowed_items() -> None:
    item_ids = ["item_car", "item_phone", "item_bike"]
    texts = [
        "ремонт автомобиля",
        "ремонт iphone смартфона",
        "ремонт велосипеда",
    ]
    index = SparseBranchIndex.fit(item_ids, texts)

    # Query is for iphone, but routing ONLY allows item_car and item_bike
    queries = ["iphone"]
    allowed = [{"item_car", "item_bike"}]

    # Fallback is 0, so item_phone MUST NOT be returned even though it matches
    results = index.retrieve_routed(queries, allowed, k=2, global_fallback_k=0)
    returned_ids = {pair[0] for pair in results[0]}

    assert "item_phone" not in returned_ids
    assert returned_ids.issubset(allowed[0])

    # If allowed set is empty, results must be empty when fallback=0
    empty_allowed = index.retrieve_routed(queries, [set()], k=2, global_fallback_k=0)
    assert empty_allowed[0] == []


def test_global_fallback_appends_expected_items_without_duplicates() -> None:
    item_ids = ["item_1", "item_2", "item_3", "item_4"]
    texts = [
        "красное яблоко",  # item_1 (in allowed, no match)
        "быстрый синий автомобиль",  # item_2 (outside allowed, MATCH)
        "красивый синий автомобиль",  # item_3 (outside allowed, MATCH)
        "зеленый дом",  # item_4 (outside allowed, no match)
    ]
    index = SparseBranchIndex.fit(item_ids, texts)

    # Allowed set contains only item_1
    queries = ["синий автомобиль"]
    allowed = [{"item_1"}]

    # k=1 routed candidate, global_fallback_k=2
    results = index.retrieve_routed(queries, allowed, k=1, global_fallback_k=2)
    candidates = results[0]

    # First candidate is from allowed (routed)
    assert len(candidates) == 3
    assert candidates[0][0] == "item_1"  # Routed candidate

    # Next two candidates are appended from global fallback
    fallback_ids = [candidates[1][0], candidates[2][0]]
    assert set(fallback_ids) == {"item_2", "item_3"}
    # Scores for fallback items are positive matches
    assert candidates[1][1] > 0.0
    assert candidates[2][1] > 0.0

    # Ensure no duplicates per query
    all_returned_ids = [pair[0] for pair in candidates]
    assert len(all_returned_ids) == len(set(all_returned_ids))


def test_tie_breaking_stability() -> None:
    # 4 items with zero relevance to query
    item_ids = ["item_z", "item_m", "item_a", "item_c"]
    texts = ["дом", "квартира", "дача", "коттедж"]
    index = SparseBranchIndex.fit(item_ids, texts)

    queries = ["космос галактика"]
    allowed = [{"item_z", "item_m", "item_a", "item_c"}]

    results = index.retrieve_routed(queries, allowed, k=3, global_fallback_k=0)
    # All scores are 0.0 -> tie broken by item_id ascending: item_a, item_c, item_m
    items_ranked = [pair[0] for pair in results[0]]
    assert items_ranked == ["item_a", "item_c", "item_m"]


def test_field_aware_sparse_index_multi_branch() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["it_1", "it_2", "it_3"],
            "item_title_raw": [
                "Apple iPhone 13 Pro",
                "Чехол силиконовый",
                "Samsung Galaxy S22",
            ],
            "item_description_raw": [
                "Отличное состояние, полный комплект коробка зарядка",
                "Прозрачный чехол для айфона 13 про",
                "Флагманский телефон в идеале",
            ],
            "item_infm_params_text": [
                "Память 256 ГБ Цвет Серый",
                "Материал силикон",
                "Память 128 ГБ Цвет Черный",
            ],
        }
    )

    index = FieldAwareSparseIndex.fit(items)

    assert index.branch_a is not None
    assert index.branch_b is not None
    assert index.branch_c is not None
    assert index.branch_d is not None

    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "search_query": ["iphone 13 pro", "чехол"],
            "search_infm_params_text": ["256 гб", ""],  # q2 has empty filter
        }
    )
    allowed = [{"it_1", "it_2", "it_3"}, {"it_1", "it_2", "it_3"}]

    multi_res = index.retrieve_multi_branch(queries, allowed, k=2)

    assert set(multi_res.keys()) == {
        "branch_a",
        "branch_b",
        "branch_c",
        "branch_d",
    }

    # Branch A (Title match): q1 top item should be it_1 (iPhone 13 Pro)
    q1_branch_a = multi_res["branch_a"][0]
    assert q1_branch_a[0][0] == "it_1"

    # Branch C (Description match): q2 query "чехол" -> it_2 description has "чехол для айфона"
    q2_branch_c = multi_res["branch_c"][1]
    assert q2_branch_c[0][0] == "it_2"

    # Branch D rule: only active when query filters are non-empty!
    # q1 has filter "256 гб" -> should retrieve it_1 which has "Память 256 ГБ"
    q1_branch_d = multi_res["branch_d"][0]
    assert len(q1_branch_d) > 0
    assert q1_branch_d[0][0] == "it_1"

    # q2 has empty filter "" -> Branch D MUST return [] (inactive)
    q2_branch_d = multi_res["branch_d"][1]
    assert q2_branch_d == []


def test_field_aware_sparse_index_save_load(tmp_path: Path) -> None:
    items = pd.DataFrame(
        {
            "item_id": ["1", "2"],
            "item_title_raw": ["велик горный", "велосипед дорожный"],
            "item_description_raw": ["скоростной 21 скорость", "удобное сиденье"],
            "item_infm_params_text": ["колеса 26", "колеса 28"],
        }
    )
    index = FieldAwareSparseIndex.fit(items)
    save_dir = tmp_path / "fa_index"
    index.save(save_dir)

    loaded = FieldAwareSparseIndex.load(save_dir)

    queries = ["горный"]
    allowed = [{"1", "2"}]

    orig_res = index.retrieve_routed(queries, allowed, k=2, branch="branch_a")
    loaded_res = loaded.retrieve_routed(queries, allowed, k=2, branch="branch_a")

    assert orig_res == loaded_res
    assert orig_res[0][0][0] == "1"


def test_module_retrieve_routed_dispatch() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["it_a", "it_b"],
            "item_title_raw": ["ноутбук игровой", "офисный ПК"],
        }
    )
    index = FieldAwareSparseIndex.fit(items)

    res = retrieve_routed(
        index,
        ["ноутбук"],
        [{"it_a", "it_b"}],
        k=1,
    )
    assert len(res) == 1
    assert res[0][0][0] == "it_a"


def test_rankings_to_dataframe() -> None:
    rankings = [
        [("it_1", 0.95), ("it_2", 0.4)],
        [("it_3", 0.8)],
    ]
    query_ids = ["q1", "q2"]
    df = rankings_to_dataframe(rankings, query_ids, source="routed_lexical")

    validate_candidates(df)
    assert len(df) == 3
    assert list(df["internal_query_id"]) == ["q1", "q1", "q2"]
    assert list(df["rank"]) == [1, 2, 1]
