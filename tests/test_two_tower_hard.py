import pandas as pd
from avito_candidate_generation.training.two_tower import prepare_hard_training_rows

def test_hard_rows_reject_duplicates():
    positives=pd.DataFrame({"internal_query_id":["q"],"item_id":["p"]})
    negatives=pd.DataFrame({"internal_query_id":["q"],"item_id":["n"]})
    assert len(prepare_hard_training_rows(positives,negatives)) == 1
