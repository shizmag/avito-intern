import pandas as pd
import pytest

from avito_candidate_generation.training.hard_negatives import (
    mine_hard_negatives,
    validate_negatives,
)


def test_mining_removes_positive_and_preserves_source():
    candidates = pd.DataFrame(
        {
            "internal_query_id": ["q", "q", "q"],
            "item_id": ["p", "n", "m"],
            "source": ["bm25", "bm25", "dense"],
            "rank": [1, 2, 1],
            "score": [3.0, 2.0, 1.0],
        }
    )
    positives = pd.DataFrame({"internal_query_id": ["q"], "item_id": ["p"]})
    result = mine_hard_negatives(candidates, positives, max_per_query=1)
    assert result.item_id.tolist() == ["n"]
    assert validate_negatives(result, positives, max_per_query=1)["status"]


def test_collision_rejected():
    with pytest.raises(ValueError):
        validate_negatives(
            pd.DataFrame(
                {"internal_query_id": ["q"], "item_id": ["p"], "source": ["x"]}
            ),
            pd.DataFrame({"internal_query_id": ["q"], "item_id": ["p"]}),
        )
