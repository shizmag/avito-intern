import pandas as pd
from avito_candidate_generation.splits import assign_item_splits, build_ground_truth


def test_item_split_is_deterministic_and_exclusive():
    a = assign_item_splits(["a", "b", "c"], seed=1)
    b = assign_item_splits(["c", "a", "b"], seed=1)
    assert a.equals(b)
    assert a["item_id"].is_unique


def test_ground_truth_follows_item_split():
    gt = build_ground_truth(
        pd.DataFrame({"internal_query_id": ["q"], "item_id": ["i"]}),
        pd.DataFrame({"item_id": ["i"], "split": ["test"]}),
    )
    assert gt.iloc[0].to_dict() == {
        "split": "test",
        "internal_query_id": "q",
        "item_id": "i",
    }
