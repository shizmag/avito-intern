import numpy as np
import pandas as pd
import pytest
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st  # type: ignore[import-not-found]

from avito_candidate_generation.candidates import (
    CandidateError,
    rank_candidates,
    validate_candidates,
)
from avito_candidate_generation.evaluation import recall_at_k
from avito_candidate_generation.retrievers.bm25 import retrieve_bm25
from avito_candidate_generation.retrievers.exact_search import exact_top_k


def test_candidate_validation_and_tie_break():
    frame = pd.DataFrame(
        {
            "internal_query_id": ["q", "q"],
            "item_id": ["b", "a"],
            "source": ["x", "x"],
            "score": [1.0, 1.0],
            "rank": [1, 2],
        }
    )
    validate_candidates(frame)
    assert rank_candidates(frame, limit=1).iloc[0].item_id == "a"
    with pytest.raises(CandidateError):
        validate_candidates(frame.assign(rank=[1, 3]))


def test_recall_uses_ground_truth_population_and_dedup():
    predictions = pd.DataFrame(
        {
            "internal_query_id": ["q1"],
            "item_id": ["i1"],
            "source": ["x"],
            "score": [1.0],
            "rank": [1],
        }
    )
    ground_truth = pd.DataFrame(
        {"internal_query_id": ["q1", "q2"], "item_id": ["i1", "i2"]}
    )
    assert recall_at_k(predictions, ground_truth, 1) == 0.5


def test_exact_search_and_bm25():
    dense = exact_top_k(
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["b", "a"],
        k=2,
        query_ids=["q"],
    )
    assert dense.iloc[0].item_id == "b"
    items = pd.DataFrame({"item_id": ["a", "b"], "text": ["red phone", "blue car"]})
    queries = pd.DataFrame({"internal_query_id": ["q"], "text": ["phone"]})
    result = retrieve_bm25(queries, items, k=2)
    assert result.iloc[0].item_id == "a"


def test_bm25_category_policy_hard_and_fallback() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "text": ["red phone", "red car", "blue phone"],
            "search_category": ["phone", "car", "phone"],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "text": ["phone", "red"],
            "search_category": ["phone", "missing"],
        }
    )
    hard = retrieve_bm25(queries, items, k=3, category_policy="hard")
    assert set(hard.loc[hard.internal_query_id == "q1", "item_id"]) == {"a", "c"}
    assert hard.loc[hard.internal_query_id == "q2"].empty
    fallback = retrieve_bm25(queries, items, k=3, category_policy="fallback")
    assert set(fallback.loc[fallback.internal_query_id == "q1", "item_id"]) == {
        "a",
        "c",
    }
    assert set(fallback.loc[fallback.internal_query_id == "q2", "item_id"]) == {
        "a",
        "b",
        "c",
    }


def test_bm25_category_policy_rejects_unknown_policy() -> None:
    items = pd.DataFrame({"item_id": ["a"], "text": ["x"], "search_category": ["c"]})
    queries = pd.DataFrame(
        {"internal_query_id": ["q"], "text": ["x"], "search_category": ["c"]}
    )
    with pytest.raises(ValueError, match="category_policy"):
        retrieve_bm25(queries, items, category_policy="location")


@given(
    query_count=st.integers(min_value=1, max_value=4),
    item_count=st.integers(min_value=1, max_value=8),
)
def test_exact_search_returns_unique_stable_top_k(
    query_count: int, item_count: int
) -> None:
    query_embeddings = np.eye(max(query_count, 1), 3, dtype=np.float32)[:query_count]
    item_embeddings = np.zeros((item_count, 3), dtype=np.float32)
    item_embeddings[:, 0] = 1.0
    result = exact_top_k(
        query_embeddings,
        item_embeddings,
        [f"i{index}" for index in range(item_count)],
        k=item_count + 3,
        query_ids=[f"q{index}" for index in range(query_count)],
    )
    assert len(result) == query_count * item_count
    unique_counts = result.groupby("internal_query_id")["item_id"].nunique().tolist()
    assert unique_counts == [item_count] * query_count


@given(
    relevant_count=st.integers(min_value=1, max_value=5),
    prediction_count=st.integers(min_value=0, max_value=8),
)
def test_recall_is_bounded(relevant_count: int, prediction_count: int) -> None:
    relevant = pd.DataFrame(
        {
            "internal_query_id": ["q"] * relevant_count,
            "item_id": [f"i{i}" for i in range(relevant_count)],
        }
    )
    predictions = pd.DataFrame(
        {
            "internal_query_id": ["q"] * prediction_count,
            "item_id": [f"i{i}" for i in range(prediction_count)],
            "source": ["x"] * prediction_count,
            "score": [1.0] * prediction_count,
            "rank": list(range(1, prediction_count + 1)),
        }
    )
    if prediction_count == 0:
        value = recall_at_k(
            pd.DataFrame(
                {
                    "internal_query_id": pd.Series(dtype=str),
                    "item_id": pd.Series(dtype=str),
                    "source": pd.Series(dtype=str),
                    "score": pd.Series(dtype=float),
                    "rank": pd.Series(dtype=int),
                }
            ),
            relevant,
            1,
        )
    else:
        value = recall_at_k(predictions, relevant, prediction_count)
    assert 0.0 <= value <= 1.0
