"""Unit tests for candidate pool union, fusion, evaluation contract, and diagnostic tools."""

import pandas as pd
import pytest

from avito_candidate_generation.fusion.candidate_pool import (
    assert_recall_monotonic,
    evaluate_rankings,
    evaluate_slices,
    incremental_union_coverage,
    oracle_candidate_coverage,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Monotonicity assertion tests
# ---------------------------------------------------------------------------


def test_assert_recall_monotonic_passes_on_valid_inputs() -> None:
    # Strictly increasing sequence
    metrics = {
        "recall@50": 0.5,
        "recall@100": 0.6,
        "recall@200": 0.7,
        "recall@500": 0.8,
        "recall@1000": 0.9,
    }
    assert_recall_monotonic(metrics)

    # Identical values across cutoffs
    flat_metrics = {
        "recall@50": 0.75,
        "recall@100": 0.75,
        "recall@200": 0.75,
        "recall@500": 0.75,
        "recall@1000": 0.75,
    }
    assert_recall_monotonic(flat_metrics)

    # Within 1e-9 tolerance
    tolerant_metrics = {
        "recall@50": 0.5000000000,
        "recall@100": 0.4999999995,  # diff is 5e-10 <= 1e-9
        "recall@200": 0.5500000000,
    }
    assert_recall_monotonic(tolerant_metrics)

    # Extra non-recall keys ignored
    mixed_metrics = {
        "mrr@10": 0.9,
        "union_coverage@50": 0.8,
        "recall@50": 0.5,
        "recall@100": 0.6,
    }
    assert_recall_monotonic(mixed_metrics)

    # Empty metrics passes
    assert_recall_monotonic({})


def test_assert_recall_monotonic_raises_on_violations() -> None:
    # Decreasing between 50 and 100
    metrics_drop_early = {
        "recall@50": 0.6,
        "recall@100": 0.5,
        "recall@200": 0.7,
        "recall@500": 0.8,
        "recall@1000": 0.9,
    }
    with pytest.raises(AssertionError, match="Recall monotonicity violated"):
        assert_recall_monotonic(metrics_drop_early)

    # Decreasing between 500 and 1000
    metrics_drop_late = {
        "recall@50": 0.5,
        "recall@100": 0.6,
        "recall@200": 0.7,
        "recall@500": 0.85,
        "recall@1000": 0.84,
    }
    with pytest.raises(AssertionError, match="Recall monotonicity violated"):
        assert_recall_monotonic(metrics_drop_late)

    # Exceeding tolerance (1.1e-9 > 1e-9)
    strict_violation = {
        "recall@50": 0.5000000000,
        "recall@100": 0.5000000000 - 1.1e-9,
    }
    with pytest.raises(AssertionError, match="Recall monotonicity violated"):
        assert_recall_monotonic(strict_violation)


# ---------------------------------------------------------------------------
# Ranking evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_rankings_recall_calculation() -> None:
    # q0: relevant are {"a", "d"}
    # q1: relevant are {"y", "w"}
    rankings: list[list[tuple[str, float]]] = [
        [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)],
        [("x", 2.0), ("y", 1.5), ("z", 1.0)],
    ]
    query_ids = ["q0", "q1"]
    relevant = {
        "q0": {"a", "d"},
        "q1": {"y", "w"},
    }

    # k=1: q0 hits "a" (1/2 = 0.5), q1 hits none (0/2 = 0.0) -> mean = 0.25
    # k=2: q0 hits "a" (1/2 = 0.5), q1 hits "y" (1/2 = 0.5) -> mean = 0.5
    # k=3: q0 hits "a" (1/2 = 0.5), q1 hits "y" (1/2 = 0.5) -> mean = 0.5
    # k=4: q0 hits "a", "d" (2/2 = 1.0), q1 hits "y" (1/2 = 0.5) -> mean = 0.75
    metrics = evaluate_rankings(rankings, query_ids, relevant, ks=(1, 2, 3, 4))

    assert metrics["recall@1"] == pytest.approx(0.25)
    assert metrics["recall@2"] == pytest.approx(0.50)
    assert metrics["recall@3"] == pytest.approx(0.50)
    assert metrics["recall@4"] == pytest.approx(0.75)


def test_evaluate_rankings_deduplication() -> None:
    # Query with duplicate items in rankings: should deduplicate preserving first occurrence
    rankings: list[list[tuple[str, float]]] = [
        [("a", 1.0), ("a", 0.8), ("b", 0.5)],
    ]
    query_ids = ["q0"]
    relevant = {"q0": {"b"}}

    # At k=1: unique top-1 is {"a"}, hit 0/1 = 0.0
    # At k=2: unique top-2 is {"a", "b"}, hit 1/1 = 1.0 (not blocked by duplicate "a")
    metrics = evaluate_rankings(rankings, query_ids, relevant, ks=(1, 2))
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0


def test_evaluate_rankings_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_rankings([], ["q0"], {})

    with pytest.raises(ValueError, match="positive"):
        evaluate_rankings([[]], ["q0"], {}, ks=(0,))


# ---------------------------------------------------------------------------
# Oracle candidate coverage tests
# ---------------------------------------------------------------------------


def test_oracle_candidate_coverage_no_truncation() -> None:
    # 2 sources, 2 queries
    # Source 0 (BM25)
    bm25: list[list[tuple[str, float]]] = [
        [("a", 1.0), ("b", 0.8), ("c", 0.6)],
        [("m", 1.0), ("n", 0.5)],
    ]
    # Source 1 (Dense)
    dense: list[list[tuple[str, float]]] = [
        [("d", 0.9), ("b", 0.7), ("e", 0.5)],
        [("p", 0.9), ("m", 0.8)],
    ]
    sources: list[list[list[tuple[str, float]]]] = [bm25, dense]
    query_ids = ["q0", "q1"]
    relevant = {
        "q0": {"b", "d", "e"},
        "q1": {"n", "p"},
    }

    # k=1:
    # q0: BM25 top-1={"a"}, Dense top-1={"d"}. Union={"a", "d"}. Hits={"d"} -> 1/3
    # q1: BM25 top-1={"m"}, Dense top-1={"p"}. Union={"m", "p"}. Hits={"p"} -> 1/2
    # mean union_coverage@1 = (1/3 + 1/2) / 2 = 5/12 ≈ 0.416667
    #
    # k=2:
    # q0: BM25 top-2={"a", "b"}, Dense top-2={"d", "b"}. Union={"a", "b", "d"}. (size 3 > k)
    #     Hits={"b", "d"} -> 2/3
    # q1: BM25 top-2={"m", "n"}, Dense top-2={"p", "m"}. Union={"m", "n", "p"}. (size 3 > k)
    #     Hits={"n", "p"} -> 2/2 = 1.0
    # mean union_coverage@2 = (2/3 + 1.0) / 2 = 5/6 ≈ 0.833333
    #
    # k=3:
    # q0: BM25 top-3={"a", "b", "c"}, Dense top-3={"d", "b", "e"}. Union={"a", "b", "c", "d", "e"}.
    #     Hits={"b", "d", "e"} -> 3/3 = 1.0
    # q1: BM25 top-3={"m", "n"}, Dense top-3={"p", "m"}. Union={"m", "n", "p"}.
    #     Hits={"n", "p"} -> 2/2 = 1.0
    # mean union_coverage@3 = (1.0 + 1.0) / 2 = 1.0
    results = oracle_candidate_coverage(sources, query_ids, relevant, ks=(1, 2, 3))

    assert results["union_coverage@1"] == pytest.approx(5.0 / 12.0)
    assert results["union_coverage@2"] == pytest.approx(5.0 / 6.0)
    assert results["union_coverage@3"] == pytest.approx(1.0)


def test_oracle_candidate_coverage_validation() -> None:
    bm25: list[list[tuple[str, float]]] = [[("a", 1.0)]]
    with pytest.raises(ValueError, match="does not match query_ids length"):
        oracle_candidate_coverage([bm25], ["q0", "q1"], {})


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion tests
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_basic_and_deduplication() -> None:
    # Query 0:
    # Source 0: [("a", 10.0), ("a", 5.0), ("b", 3.0)]  -> duplicate "a" should be ignored
    # Source 1: [("b", 8.0), ("c", 4.0)]
    s0: list[list[tuple[str, float]]] = [[("a", 10.0), ("a", 5.0), ("b", 3.0)]]
    s1: list[list[tuple[str, float]]] = [[("b", 8.0), ("c", 4.0)]]

    # c = 60.0
    # Source 0: rank(a)=1 -> 1/61, rank(b)=2 -> 1/62
    # Source 1: rank(b)=1 -> 1/61, rank(c)=2 -> 1/62
    # Total scores:
    # b: 1/62 + 1/61 = 123 / 3782 ≈ 0.032522
    # a: 1/61 ≈ 0.016393
    # c: 1/62 ≈ 0.016129
    fused = reciprocal_rank_fusion([s0, s1], c=60.0, limit=10)
    assert len(fused) == 1
    ranked = fused[0]
    items = [item for item, _ in ranked]
    assert items == ["b", "a", "c"]
    assert ranked[0][1] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert ranked[1][1] == pytest.approx(1.0 / 61.0)
    assert ranked[2][1] == pytest.approx(1.0 / 62.0)


def test_reciprocal_rank_fusion_weighted() -> None:
    s0: list[list[tuple[str, float]]] = [[("a", 1.0), ("b", 0.5)]]
    s1: list[list[tuple[str, float]]] = [[("b", 1.0), ("c", 0.5)]]

    # Give source 0 weight 3.0, source 1 weight 1.0
    # a: 3.0 / 61 ≈ 0.04918
    # b: 3.0 / 62 + 1.0 / 61 = 3/62 + 1/61 ≈ 0.048387 + 0.016393 = 0.06478
    # c: 1.0 / 62 ≈ 0.016129
    fused = reciprocal_rank_fusion([s0, s1], weights=[3.0, 1.0], c=60.0, limit=2)
    assert len(fused[0]) == 2
    assert fused[0][0][0] == "b"
    assert fused[0][1][0] == "a"


def test_reciprocal_rank_fusion_stable_tie_breaking() -> None:
    # Both items have identical scores across 2 sources
    # s0: "z_item" rank 1, "a_item" rank 2
    # s1: "a_item" rank 1, "z_item" rank 2
    # Both have total score = 1/61 + 1/62
    s0: list[list[tuple[str, float]]] = [[("z_item", 10.0), ("a_item", 9.0)]]
    s1: list[list[tuple[str, float]]] = [[("a_item", 10.0), ("z_item", 9.0)]]

    fused = reciprocal_rank_fusion([s0, s1], c=60.0, limit=10)
    # Tie breaking: score desc, item_id asc -> "a_item" before "z_item"
    assert [x[0] for x in fused[0]] == ["a_item", "z_item"]


def test_reciprocal_rank_fusion_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([[[("a", 1.0)]]], c=0.0)

    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([[[("a", 1.0)]]], limit=-1)

    with pytest.raises(ValueError, match="match number of sources"):
        reciprocal_rank_fusion([[[("a", 1.0)]]], weights=[1.0, 2.0])


# ---------------------------------------------------------------------------
# Incremental union coverage tests
# ---------------------------------------------------------------------------


def test_incremental_union_coverage() -> None:
    # Base: BM25 retrieves "a" (hit 1/2 = 0.5)
    base_bm25: list[list[tuple[str, float]]] = [[("a", 1.0)]]
    # New branch: Historical intent retrieves "b" (recovering remaining positive)
    new_hist: list[list[tuple[str, float]]] = [[("b", 1.0)]]
    # Redundant branch: retrieves only "a"
    redundant: list[list[tuple[str, float]]] = [[("a", 0.5)]]

    query_ids = ["q0"]
    relevant = {"q0": {"a", "b"}}

    # Adding new_hist increases coverage from 0.5 to 1.0 (+0.5)
    delta_hist = incremental_union_coverage(
        [base_bm25], new_hist, query_ids, relevant, k=1
    )
    assert delta_hist == pytest.approx(0.5)

    # Adding redundant source yields 0.0 incremental coverage
    delta_redundant = incremental_union_coverage(
        [base_bm25], redundant, query_ids, relevant, k=1
    )
    assert delta_redundant == pytest.approx(0.0)

    # Base is empty: incremental coverage is coverage of new branch itself
    delta_empty_base = incremental_union_coverage(
        [], new_hist, query_ids, relevant, k=1
    )
    assert delta_empty_base == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Sliced evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_slices_boolean_and_categorical() -> None:
    query_ids = ["q0", "q1", "q2", "q3"]
    query_df = pd.DataFrame(
        {
            "internal_query_id": query_ids,
            "is_short": [True, False, True, False],
            "category": ["auto", "auto", "realty", "realty"],
        }
    )
    rankings: list[list[tuple[str, float]]] = [
        [("item_0", 1.0)],
        [("item_1", 1.0)],
        [("item_2", 1.0)],
        [("item_3", 1.0)],
    ]
    # Ground truth: only q0 and q1 hit their targets
    relevant = {
        "q0": {"item_0"},
        "q1": {"item_1"},
        "q2": {"target_2"},  # missed
        "q3": {"target_3"},  # missed
    }

    slices = {
        "short_queries": query_df["is_short"],
        "category": query_df["category"],
    }

    results = evaluate_slices(
        rankings, query_ids, query_df, relevant, slices, ks=(1, 5)
    )

    # short_queries includes q0 (hit) and q2 (miss): recall@1 = 0.5
    assert "short_queries" in results
    assert results["short_queries"]["recall@1"] == pytest.approx(0.5)

    # category "auto" includes q0 (hit) and q1 (hit): recall@1 = 1.0
    assert "category_auto" in results
    assert results["category_auto"]["recall@1"] == pytest.approx(1.0)

    # category "realty" includes q2 (miss) and q3 (miss): recall@1 = 0.0
    assert "category_realty" in results
    assert results["category_realty"]["recall@1"] == pytest.approx(0.0)


def test_evaluate_slices_empty_slice() -> None:
    query_ids = ["q0"]
    query_df = pd.DataFrame({"internal_query_id": ["q0"], "flag": [False]})
    rankings: list[list[tuple[str, float]]] = [[("a", 1.0)]]
    relevant = {"q0": {"a"}}

    slices = {"empty_slice": query_df["flag"]}
    results = evaluate_slices(rankings, query_ids, query_df, relevant, slices, ks=(10,))

    assert results["empty_slice"]["recall@10"] == 0.0
