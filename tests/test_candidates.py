import pandas as pd

from avito_candidate_generation.candidates import validate_candidates


def test_candidate_schema():
    frame = pd.DataFrame(
        {
            "internal_query_id": ["q"],
            "item_id": ["i"],
            "source": ["x"],
            "score": [1.0],
            "rank": [1],
        }
    )
    validate_candidates(frame)
