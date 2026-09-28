import pandas as pd

from avito_candidate_generation.inference import rank_predictions


def test_rank_predictions_contiguous():
    result = rank_predictions(
        pd.DataFrame(
            {"query_id": ["q", "q"], "item_id": ["b", "a"], "final_score": [1.0, 1.0]}
        )
    )
    assert result["rank"].tolist() == [1, 2]
