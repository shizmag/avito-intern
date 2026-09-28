"""Unit tests for microcategory routing module."""

from __future__ import annotations

import pandas as pd
import pytest

from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
    evaluate_microcategory_routing,
)


def test_exact_posterior_fitting_and_deterministic_predictions() -> None:
    """Test exact posterior fitting, probabilities, and deterministic tie-breaking."""
    # Data with ties:
    # "car" -> "cat_20" (count 2), "cat_10" (count 2). Tie-break: "cat_10" < "cat_20".
    # "phone" -> "cat_1" (count 3), "cat_2" (count 1). Count desc: "cat_1" then "cat_2".
    # "laptop" -> "cat_50" (count 1).
    queries = [
        "car",
        "car",
        "car",
        "car",
        "phone",
        "phone",
        "phone",
        "phone",
        "laptop",
    ]
    categories = [
        "cat_20",
        "cat_20",
        "cat_10",
        "cat_10",
        "cat_1",
        "cat_1",
        "cat_1",
        "cat_2",
        "cat_50",
    ]

    predictor = ExactPosteriorPredictor()
    predictor.fit(queries, categories)

    # 1. Deterministic tie-breaking by count desc, then category_id asc
    preds_car = predictor.predict_top_k(["car"], k=2)[0]
    assert preds_car == ["cat_10", "cat_20"]

    # 2. Ordered by count desc
    preds_phone = predictor.predict_top_k(["phone"], k=2)[0]
    assert preds_phone == ["cat_1", "cat_2"]

    # 3. Probabilities for seen queries
    probas_phone = predictor.predict_proba(["phone"], top_k=2)[0]
    assert probas_phone == [("cat_1", 0.75), ("cat_2", 0.25)]

    probas_car = predictor.predict_proba(["car"], top_k=2)[0]
    assert probas_car == [("cat_10", 0.5), ("cat_20", 0.5)]

    # 4. Fallback to global prior for unseen query
    # Global counts:
    # cat_1: 3
    # cat_10: 2
    # cat_20: 2 (tie with cat_10 -> cat_10 first)
    # cat_2: 1
    # cat_50: 1 (tie with cat_2 -> cat_2 first)
    preds_unseen = predictor.predict_top_k(["unseen_query"], k=4)[0]
    assert preds_unseen == ["cat_1", "cat_10", "cat_20", "cat_2"]

    probas_unseen = predictor.predict_proba(["unseen_query"], top_k=3)[0]
    total_samples = 9
    assert probas_unseen[0] == ("cat_1", pytest.approx(3 / total_samples))
    assert probas_unseen[1] == ("cat_10", pytest.approx(2 / total_samples))
    assert probas_unseen[2] == ("cat_20", pytest.approx(2 / total_samples))

    # 5. Support and is_seen methods
    assert predictor.is_seen("car") is True
    assert predictor.is_seen("unseen_query") is False
    assert predictor.support("phone") == 4
    assert predictor.support("unseen_query") == 0

    # 6. DataFrame input support
    df = pd.DataFrame({"search_query": queries, "item_microcat_id": categories})
    df_predictor = ExactPosteriorPredictor()
    df_predictor.fit(df)
    assert df_predictor.predict_top_k(["car"], k=2)[0] == ["cat_10", "cat_20"]


def test_exact_posterior_edge_cases() -> None:
    """Test validation errors for empty data and invalid parameters."""
    predictor = ExactPosteriorPredictor()
    with pytest.raises(ValueError, match="not fitted"):
        predictor.predict_top_k(["query"], k=1)

    with pytest.raises(ValueError, match="empty"):
        predictor.fit([], [])

    with pytest.raises(ValueError, match="mismatch"):
        predictor.fit(["a", "b"], ["cat1"])

    predictor.fit(["a"], ["cat1"])
    with pytest.raises(ValueError, match="k must be positive"):
        predictor.predict_top_k(["a"], k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        predictor.predict_proba(["a"], top_k=0)


def test_generalizing_classifier_fitting_and_unseen_queries() -> None:
    """Test generalizing classifier with TF-IDF n-grams predicting on unseen queries."""
    train_queries = [
        "айфон 13 про",
        "самсунг гэлакси s21",
        "макбук про 14",
        "асус рог ноутбук",
        "баня из бруса",
        "сауна кедр сруб",
    ]
    infm_params = [
        "память 128 гб",
        "память 256 гб",
        "ноутбук m1 16 гб",
        "ноутбук rtx 3060",
        "строительство под ключ",
        "строительство отделка",
    ]
    search_categories = [
        "телефоны",
        "телефоны",
        "компьютеры",
        "компьютеры",
        "услуги",
        "услуги",
    ]
    target_categories = [
        "cat_phones",
        "cat_phones",
        "cat_laptops",
        "cat_laptops",
        "cat_baths",
        "cat_baths",
    ]

    clf_predictor = GeneralizingClassifierPredictor(seed=42, alpha=1e-4)
    clf_predictor.fit(
        train_queries,
        target_categories,
        infm_params=infm_params,
        search_categories=search_categories,
    )

    # Test on unseen queries that share words and char n-grams
    test_queries = ["айфон 12", "макбук эйр", "баня бочка"]
    top_preds = clf_predictor.predict_top_k(test_queries, k=1)
    assert top_preds[0][0] == "cat_phones"
    assert top_preds[1][0] == "cat_laptops"
    assert top_preds[2][0] == "cat_baths"

    # Test predict_proba outputs valid distribution
    probas = clf_predictor.predict_proba(test_queries, top_k=3)
    for query_probas in probas:
        assert len(query_probas) == 3
        prob_sum = sum(prob for _, prob in query_probas)
        assert prob_sum == pytest.approx(1.0, rel=1e-3)
        # Check sorted descending
        for j in range(len(query_probas) - 1):
            assert query_probas[j][1] >= query_probas[j + 1][1]

    # Test single-class fallback
    single_clf = GeneralizingClassifierPredictor()
    single_clf.fit(["item1", "item2"], ["single_cat", "single_cat"])
    assert single_clf.predict_top_k(["test"], k=1) == [["single_cat"]]
    assert single_clf.predict_proba(["test"], top_k=1) == [[("single_cat", 1.0)]]


def test_hybrid_predictor_behavior_on_seen_vs_unseen() -> None:
    """Test hybrid predictor routing based on support threshold, fallback and blending."""
    exact = ExactPosteriorPredictor().fit(
        ["seen_frequent", "seen_frequent", "seen_rare"],
        ["cat_A", "cat_A", "cat_A"],
    )
    # Generalizing predictor predicts cat_B for query containing 'beta'
    gen = GeneralizingClassifierPredictor(seed=42).fit(
        ["sample_alpha", "sample_beta"],
        ["cat_A", "cat_B"],
    )

    # 1. Fallback strategy with min_count = 2
    hybrid_fallback = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=gen,
        min_count=2,
        blend_strategy="fallback",
    )

    # "seen_frequent" has count 2 >= 2 -> uses exact posterior -> cat_A
    assert hybrid_fallback.predict_top_k(["seen_frequent"], k=1)[0] == ["cat_A"]

    # "sample_beta" is unseen in exact predictor (count 0) -> uses generalizing classifier -> cat_B
    assert hybrid_fallback.predict_top_k(["sample_beta"], k=1)[0] == ["cat_B"]

    # "seen_rare" has count 1 < 2 -> falls back to generalizing model
    # When query has 'beta' feature, generalizing classifier predicts cat_B
    rare_fallback_pred = hybrid_fallback.predict_top_k(
        ["seen_rare"], k=1, infm_params=["sample_beta"]
    )[0]
    assert rare_fallback_pred == ["cat_B"]

    # 2. Blend strategy with min_count = 2
    hybrid_blend = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=gen,
        min_count=2,
        blend_strategy="blend",
    )

    # For seen_rare (count 1 < 2), posterior weight = 1/2, generalizing weight = 1/2
    probas = hybrid_blend.predict_proba(
        ["seen_rare"], top_k=2, infm_params=["sample_beta"]
    )[0]
    assert len(probas) == 2
    cats = {cat for cat, _ in probas}
    assert cats == {"cat_A", "cat_B"}
    # Blended prob for cat_A should be 0.5 * 1.0 + 0.5 * P_gen(cat_A) > 0.5
    prob_a = next(prob for cat, prob in probas if cat == "cat_A")
    assert prob_a > 0.5


def test_multi_label_positive_support_in_evaluation() -> None:
    """Test evaluation function when queries have multiple ground-truth items across categories."""
    item_to_microcat = {
        "item_phone_1": "cat_phones",
        "item_phone_2": "cat_phones",
        "item_case_1": "cat_cases",
        "item_bike_1": "cat_bikes",
    }

    # Query with multiple ground truth items across DIFFERENT microcategories
    query_ids = ["q_bundle"]
    query_texts = ["чехол для телефона"]
    relevant_items = {"q_bundle": {"item_phone_1", "item_case_1"}}
    seen_queries = {"чехол для телефона"}
    benchmark_corpus_size = 4

    # Case A: Prediction contains only cat_phones
    # Routed items: item_phone_1, item_phone_2 (size 2)
    # Expected items: item_phone_1, item_case_1
    # Contained: item_phone_1 (1 out of 2 -> 0.5 containment)
    # Category hit: cat_phones is in {cat_phones, cat_cases} -> 1.0 hit recall
    preds_a = {1: [["cat_phones"]]}
    eval_a = evaluate_microcategory_routing(
        predictions=preds_a,
        query_ids=query_ids,
        query_texts=query_texts,
        relevant_items=relevant_items,
        item_to_microcat=item_to_microcat,
        seen_queries=seen_queries,
        benchmark_corpus_size=benchmark_corpus_size,
        top_ks=(1,),
    )
    assert eval_a[1]["category_hit_recall"] == 1.0
    assert eval_a[1]["positive_item_containment"] == 0.5
    assert eval_a[1]["corpus_size_mean"] == 2.0
    assert eval_a[1]["overall"]["positive_item_containment"] == 0.5
    assert eval_a[1]["seen_query"]["positive_item_containment"] == 0.5
    assert eval_a[1]["unseen_query"]["n_queries"] == 0

    # Case B: Prediction contains both cat_phones and cat_cases
    # Routed items: item_phone_1, item_phone_2, item_case_1 (size 3)
    # Contained: item_phone_1, item_case_1 (2 out of 2 -> 1.0 containment)
    preds_b = {2: [["cat_phones", "cat_cases"]]}
    eval_b = evaluate_microcategory_routing(
        predictions=preds_b,
        query_ids=query_ids,
        query_texts=query_texts,
        relevant_items=relevant_items,
        item_to_microcat=item_to_microcat,
        seen_queries=seen_queries,
        benchmark_corpus_size=benchmark_corpus_size,
        top_ks=(2,),
    )
    assert eval_b[2]["category_hit_recall"] == 1.0
    assert eval_b[2]["positive_item_containment"] == 1.0
    assert eval_b[2]["corpus_size_mean"] == 3.0

    # Case C: Prediction contains only irrelevant category
    preds_c = {1: [["cat_bikes"]]}
    eval_c = evaluate_microcategory_routing(
        predictions=preds_c,
        query_ids=query_ids,
        query_texts=query_texts,
        relevant_items=relevant_items,
        item_to_microcat=item_to_microcat,
        seen_queries=seen_queries,
        benchmark_corpus_size=benchmark_corpus_size,
        top_ks=(1,),
    )
    assert eval_c[1]["category_hit_recall"] == 0.0
    assert eval_c[1]["positive_item_containment"] == 0.0
    assert eval_c[1]["corpus_size_mean"] == 1.0


def test_metric_calculations_containment_hit_recall_percentiles() -> None:
    """Test detailed metric calculations including percentiles and seen/unseen breakdown."""
    # 20 queries: 10 seen, 10 unseen
    query_ids = [f"q_{i}" for i in range(20)]
    query_texts = [f"query_{i}" for i in range(20)]
    seen_queries = {f"query_{i}" for i in range(10)}

    # Map items to 20 categories
    item_to_microcat: dict[str, str] = {}
    for i in range(20):
        # Category i contains (i + 1) items
        for j in range(i + 1):
            item_to_microcat[f"item_{i}_{j}"] = f"cat_{i}"

    total_items = len(item_to_microcat)
    # Each query q_i has relevant item f"item_{i}_0" in cat_i
    relevant_items = {f"q_{i}": {f"item_{i}_0"} for i in range(20)}

    # Predictions for K=1:
    # For queries 0..9 (seen): first 5 correct (cat_i), next 5 incorrect (cat_other)
    # For queries 10..19 (unseen): first 5 correct (cat_i), next 5 incorrect (cat_other)
    preds_k1: list[list[str]] = []
    for i in range(20):
        if i % 10 < 5:
            preds_k1.append([f"cat_{i}"])
        else:
            preds_k1.append(["cat_nonexistent"])

    predictions = {1: preds_k1}

    report = evaluate_microcategory_routing(
        predictions=predictions,
        query_ids=query_ids,
        query_texts=query_texts,
        relevant_items=relevant_items,
        item_to_microcat=item_to_microcat,
        seen_queries=seen_queries,
        benchmark_corpus_size=total_items,
        top_ks=(1,),
    )

    assert report["n_queries"] == 20
    assert report["seen_queries_count"] == 10
    assert report["unseen_queries_count"] == 10

    # Overall: 10 hits out of 20 -> 0.5
    assert report[1]["category_hit_recall"] == 0.5
    assert report[1]["positive_item_containment"] == 0.5

    # Breakdown seen vs unseen: each has 5 hits out of 10 -> 0.5
    assert report[1]["seen_query"]["category_hit_recall"] == 0.5
    assert report[1]["seen_query"]["positive_item_containment"] == 0.5
    assert report[1]["seen_query"]["n_queries"] == 10

    assert report[1]["unseen_query"]["category_hit_recall"] == 0.5
    assert report[1]["unseen_query"]["positive_item_containment"] == 0.5
    assert report[1]["unseen_query"]["n_queries"] == 10

    # Verify percentiles on corpus sizes
    # Sizes for the 20 queries:
    # Correct queries i in {0,1,2,3,4, 10,11,12,13,14} -> sizes (i+1) = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    # Incorrect queries have cat_nonexistent -> 0 items (10 times)
    # All 20 sizes sorted:
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    sorted_sizes = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        11,
        12,
        13,
        14,
        15,
    ]
    expected_mean = sum(sorted_sizes) / 20
    expected_median = sorted_sizes[len(sorted_sizes) // 2]  # index 10 -> 1
    p95_idx = min(len(sorted_sizes) - 1, 20 * 95 // 100)  # index 19 -> 15
    expected_p95 = sorted_sizes[p95_idx]
    expected_max = 15

    assert report[1]["corpus_size_mean"] == expected_mean
    assert report[1]["corpus_size_median"] == expected_median
    assert report[1]["corpus_size_p95"] == expected_p95
    assert report[1]["corpus_size_max"] == expected_max
