import pandas as pd

from avito_candidate_generation.fusion.catboost import (
    load_selector,
    save_selector,
    score_selector,
    train_selector,
)
from avito_candidate_generation.oof import build_oof_table, verify_oof


def test_oof_verify_and_selector(tmp_path):
    features = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q1", "q2"],
            "item_id": ["a", "b", "c"],
            "x": [1.0, 0.0, 2.0],
        }
    )
    gt = pd.DataFrame({"internal_query_id": ["q1"], "item_id": ["a"]})
    table = pd.concat(
        [build_oof_table(features, gt, fold=f) for f in range(3)], ignore_index=True
    )
    assert verify_oof(table, folds=3)["status"]
    model = train_selector(table)
    path = save_selector(model, tmp_path / "model.json")
    loaded = load_selector(path)
    assert model["algorithm"] == "CatBoostClassifier"
    assert (tmp_path / "model.cbm").is_file()
    assert score_selector(table, loaded).notna().all()
    assert (
        score_selector(table, model).tolist() == score_selector(table, loaded).tolist()
    )
