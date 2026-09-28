import pandas as pd
import pytest

from avito_candidate_generation.research import (
    SparseRetrievalIndex,
    build_research_validation,
    canonical_numbers,
    context_folds,
    ground_truth_map,
    normalize_research_text,
    query_text,
    ranking_metrics,
    union_rankings,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "search_query": ["баня", "баня", "автоподбор", "друга"],
            "search_location_id": [1, 1, 1, 2],
            "search_is_delivery_search": [0, 0, 0, 1],
            "search_infm_params_text": ["", "", "год 2020", ""],
            "search_category": [114, 114, 114, 0],
            "item_id": ["a" * 16, "b" * 16, "c" * 16, "d" * 16],
            "item_microcat_id": [1, 1, 2, 3],
        }
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["a" * 16, "b" * 16, "c" * 16, "z" * 16],
            "item_microcat_id": [1, 1, 2, 4],
        }
    )


def test_normalization_and_numbers_are_deterministic() -> None:
    assert normalize_research_text(" ЁЛКА  ") == "елка"
    assert canonical_numbers("Год 2,0 и 10 кг") == {"2.0", "10кг"}


def test_context_folds_are_deterministic() -> None:
    assert context_folds(["b", "a"], seed=42).equals(
        context_folds(["a", "b"], seed=42)
    )


def test_research_validation_uses_full_context_and_full_benchmark_corpus() -> None:
    from avito_candidate_generation.research import context_frame

    contexts = context_frame(_train())
    banana_id = str(contexts.iloc[0]["internal_query_id"])
    banana_fold = int(
        context_folds(contexts["internal_query_id"], seed=42, folds=2)
        .set_index("internal_query_id")
        .loc[banana_id, "fold"]
    )
    validation = build_research_validation(
        _train(), _items(), seed=42, train_fold=1 - banana_fold, folds=2
    )
    assert len(validation.benchmark_items) == 4
    assert {"a" * 16, "b" * 16}.issubset(
        set(validation.ground_truth["item_id"])
    )
    assert validation.protocol["unknown_interactions_are_not_negatives"] is True
    assert validation.ground_truth.duplicated(["internal_query_id", "item_id"]).sum() == 0


def test_query_text_uses_only_query_side_fields() -> None:
    frame = _train().iloc[[0]].copy()
    assert "баня" in query_text(frame, include_location=False)[0]
    assert "location_" not in query_text(frame, include_location=False)[0]


def test_sparse_index_and_metrics() -> None:
    index = SparseRetrievalIndex.fit(
        ["a" * 16, "b" * 16], ["баня дрова", "автоподбор"], name="title"
    )
    rankings = index.top_k(["баня"], k=2)
    assert rankings[0][0][0] == "a" * 16
    metrics = ranking_metrics(rankings, ["q"], {"q": {"a" * 16}}, ks=(1, 2, 2))
    assert metrics == {"recall@1": 1.0, "recall@2": 1.0}


def test_recall_is_monotonic_when_candidate_k_duplicates_requested_k() -> None:
    from avito_candidate_generation.research import ranking_metrics

    ranking = [[("a" * 16, 1.0), ("b" * 16, 0.5)]]
    assert ranking_metrics(ranking, ["q"], {"q": {"a" * 16}}, ks=(50, 2, 50)) == {
        "recall@50": 1.0,
        "recall@2": 1.0,
    }


def test_union_is_not_truncated_before_requested_limit() -> None:
    first = [[("a" * 16, 1.0), ("b" * 16, 0.5)]]
    second = [[("c" * 16, 1.0), ("d" * 16, 0.5)]]
    union = union_rankings([first, second], limit=4)
    assert {item for item, _ in union[0]} == {"a" * 16, "b" * 16, "c" * 16, "d" * 16}


def test_ground_truth_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ground_truth_map(pd.DataFrame({"internal_query_id": ["q", "q"], "item_id": ["i", "i"]}))
