import numpy as np
import pandas as pd
import pytest
from avito_candidate_generation.candidates import (
    CandidateError,
    rank_candidates,
    validate_candidates,
)
from avito_candidate_generation.evaluation import recall_at_k
from avito_candidate_generation.retrievers.exact_search import exact_top_k
from avito_candidate_generation.retrievers.bm25 import retrieve_bm25


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
