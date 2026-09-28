"""Unit tests for RoutedDenseE5Retriever and E5QueryEncoder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from avito_candidate_generation.retrievers.routed_e5 import (
    E5QueryEncoder,
    RoutedDenseE5Retriever,
)


@pytest.fixture
def synthetic_retrieval_data() -> tuple[
    list[str], np.ndarray, dict[str, int], np.ndarray
]:
    """Generate 100 items x 16 dims and 5 queries x 16 dims normalized synthetic embeddings."""
    rng = np.random.default_rng(42)
    n_items = 100
    n_queries = 5
    dims = 16

    raw_items = rng.standard_normal((n_items, dims)).astype(np.float32)
    item_norms = np.linalg.norm(raw_items, axis=1, keepdims=True)
    item_embeddings = raw_items / item_norms

    item_ids = [f"item_{i:03d}" for i in range(n_items)]
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    raw_queries = rng.standard_normal((n_queries, dims)).astype(np.float32)
    query_norms = np.linalg.norm(raw_queries, axis=1, keepdims=True)
    query_embeddings = raw_queries / query_norms

    return item_ids, item_embeddings, item_id_to_idx, query_embeddings


def test_global_retrieval_matches_exact_dot_product_sort(
    synthetic_retrieval_data: tuple[list[str], np.ndarray, dict[str, int], np.ndarray],
) -> None:
    """Global retrieval matches exact dot product argpartition/sort."""
    item_ids, item_embeddings, item_id_to_idx, query_embeddings = (
        synthetic_retrieval_data
    )
    retriever = RoutedDenseE5Retriever(
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        item_id_to_idx=item_id_to_idx,
    )

    for k in (1, 5, 20, 100, 120):
        actual = retriever.retrieve_global(query_embeddings, k=k, batch_size=2)
        assert len(actual) == len(query_embeddings)

        # Baseline brute-force calculation
        all_scores = query_embeddings @ item_embeddings.T
        for q_idx in range(len(query_embeddings)):
            scores = all_scores[q_idx]
            order = sorted(
                range(len(item_ids)),
                key=lambda idx: (-float(scores[idx]), item_ids[idx]),
            )[:k]
            expected = [(item_ids[idx], float(scores[idx])) for idx in order]

            actual_q = actual[q_idx]
            assert len(actual_q) == len(expected)
            assert [x[0] for x in actual_q] == [x[0] for x in expected]
            np.testing.assert_allclose(
                [x[1] for x in actual_q],
                [x[1] for x in expected],
                rtol=1e-5,
                atol=1e-6,
            )


def test_routed_retrieval_returns_only_allowed_items(
    synthetic_retrieval_data: tuple[list[str], np.ndarray, dict[str, int], np.ndarray],
) -> None:
    """Routed retrieval returns only allowed items and scores match subset dot product."""
    item_ids, item_embeddings, item_id_to_idx, query_embeddings = (
        synthetic_retrieval_data
    )
    retriever = RoutedDenseE5Retriever(
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        item_id_to_idx=item_id_to_idx,
    )

    allowed_sets: list[set[str]] = [
        {f"item_{i:03d}" for i in range(15)},
        {"item_005", "item_010", "item_020"},
        {f"item_{i:03d}" for i in range(25, 65)},
        set(),
        {"item_001", "item_002", "unknown_item_999"},
    ]

    k = 10
    actual = retriever.retrieve_routed(
        query_embeddings, allowed_sets, k=k, global_fallback_k=0
    )
    assert len(actual) == len(query_embeddings)

    for q_idx, allowed in enumerate(allowed_sets):
        results_q = actual[q_idx]
        returned_ids = [item_id for item_id, _ in results_q]

        # All returned IDs must be in the query's allowed set
        assert set(returned_ids).issubset(allowed)
        assert "unknown_item_999" not in returned_ids

        valid_in_index = [it for it in allowed if it in item_id_to_idx]
        assert len(results_q) == min(k, len(valid_in_index))

        # Check exact subset dot product scores
        if valid_in_index:
            sub_indices = [item_id_to_idx[it] for it in valid_in_index]
            sub_scores = (
                item_embeddings[np.array(sub_indices, dtype=np.int64)]
                @ query_embeddings[q_idx]
            )
            expected_order = sorted(
                range(len(valid_in_index)),
                key=lambda idx: (-float(sub_scores[idx]), valid_in_index[idx]),
            )[:k]
            expected_ids = [valid_in_index[i] for i in expected_order]
            expected_scores = [float(sub_scores[i]) for i in expected_order]

            assert returned_ids == expected_ids
            np.testing.assert_allclose(
                [s for _, s in results_q],
                expected_scores,
                rtol=1e-5,
                atol=1e-6,
            )


def test_global_fallback_adds_global_items_not_in_routed_set(
    synthetic_retrieval_data: tuple[list[str], np.ndarray, dict[str, int], np.ndarray],
) -> None:
    """Global fallback adds global items not in routed set, deduplicated and sorted."""
    item_ids, item_embeddings, item_id_to_idx, query_embeddings = (
        synthetic_retrieval_data
    )
    retriever = RoutedDenseE5Retriever(
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        item_id_to_idx=item_id_to_idx,
    )

    # Restrict allowed set to 2 items
    allowed_sets = [{"item_001", "item_002"}]
    single_query = query_embeddings[:1]

    # Global top 5 items
    global_top_5 = retriever.retrieve_global(single_query, k=5)[0]
    global_ids_5 = [it for it, _ in global_top_5]

    # Routed with k=2 and global_fallback_k=5
    routed_with_fallback = retriever.retrieve_routed(
        single_query, allowed_sets, k=2, global_fallback_k=5
    )[0]

    returned_ids = [it for it, _ in routed_with_fallback]
    routed_set = {"item_001", "item_002"}

    # Expected items: routed items + global items not in routed set
    expected_fallback_items = [it for it in global_ids_5 if it not in routed_set]
    assert len(expected_fallback_items) > 0

    # Ensure all fallback items not in routed set are present
    for item_id in expected_fallback_items:
        assert item_id in returned_ids

    # Ensure no duplicates
    assert len(returned_ids) == len(set(returned_ids))

    # Ensure total count matches routed + new fallback items
    assert len(returned_ids) == len(routed_set) + len(expected_fallback_items)


def test_monotonicity_and_deduplication(
    synthetic_retrieval_data: tuple[list[str], np.ndarray, dict[str, int], np.ndarray],
) -> None:
    """Scores are monotonic desc, item_ids are unique, and tie-breaking is item_id asc."""
    item_ids, item_embeddings, item_id_to_idx, query_embeddings = (
        synthetic_retrieval_data
    )
    retriever = RoutedDenseE5Retriever(
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        item_id_to_idx=item_id_to_idx,
    )

    allowed_sets: list[set[str]] = [
        {f"item_{i:03d}" for i in range(10)},
        {f"item_{i:03d}" for i in range(5, 25)},
        {"item_099"},
        set(),
        {f"item_{i:03d}" for i in range(100)},
    ]

    for fallback_k in (0, 3, 10):
        results = retriever.retrieve_routed(
            query_embeddings, allowed_sets, k=5, global_fallback_k=fallback_k
        )
        for q_res in results:
            # 1. Deduplication
            ids = [it for it, _ in q_res]
            assert len(ids) == len(set(ids))

            # 2. Monotonicity
            scores = [sc for _, sc in q_res]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]


def test_tie_breaking_consistency() -> None:
    """Verify deterministic tie-breaking: score desc, item_id asc."""
    # Create 3 items with identical embeddings
    item_ids = ["item_c", "item_a", "item_b"]
    item_embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    retriever = RoutedDenseE5Retriever(item_ids, item_embeddings)
    results = retriever.retrieve_global(query_embeddings, k=3)[0]

    # All have score 1.0, must be sorted alphabetically by item_id
    assert [it for it, _ in results] == ["item_a", "item_b", "item_c"]
    assert [sc for _, sc in results] == [1.0, 1.0, 1.0]

    # Now test with routed fallback tie-breaking
    routed = retriever.retrieve_routed(
        query_embeddings, [{"item_c"}], k=1, global_fallback_k=3
    )[0]
    # Routed has item_c (1.0). Fallback adds item_a (1.0) and item_b (1.0).
    # Combined must be sorted alphabetically by item_id: item_a, item_b, item_c
    assert [it for it, _ in routed] == ["item_a", "item_b", "item_c"]


class MockTokenizer:
    """Mock HuggingFace tokenizer tracking calls and prefixes."""

    def __init__(self) -> None:
        self.call_history: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        self.call_history.append(list(texts))
        batch_size = len(texts)
        seq_len = 8
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MockModel(torch.nn.Module):
    """Mock Transformer model outputting deterministic last hidden states."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.call_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        batch_size, seq_len = input_ids.shape
        # Return state where sum across tokens is non-zero
        states = torch.ones((batch_size, seq_len, self.hidden_dim), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=states)


def test_encoder_prefix_normalization_and_caching() -> None:
    """Verify E5QueryEncoder adds 'query: ' prefix, normalizes L2, and caches embeddings."""
    mock_tok = MockTokenizer()
    mock_mod = MockModel(hidden_dim=16)

    encoder = E5QueryEncoder(
        device="cpu",
        model=mock_mod,
        tokenizer=mock_tok,
    )

    texts = ["apple iphone", "query: samsung galaxy"]
    embeddings = encoder.encode_queries(texts, batch_size=2)

    # 1. Prefix verification
    assert mock_tok.call_history[0] == ["query: apple iphone", "query: samsung galaxy"]

    # 2. L2 normalization verification
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # 3. Cache verification
    assert "apple iphone" in encoder.cache
    assert "query: samsung galaxy" in encoder.cache
    assert mock_mod.call_count == 1

    # Encode with one cached and one new query
    mock_tok.call_history.clear()
    more_texts = ["apple iphone", "google pixel"]
    more_embeddings = encoder.encode_queries(more_texts, batch_size=2)

    # Only "google pixel" should have been passed to the model
    assert mock_tok.call_history[0] == ["query: google pixel"]
    assert mock_mod.call_count == 2
    assert "google pixel" in encoder.cache
    assert len(more_embeddings) == 2
    # Verify cached embedding matches
    np.testing.assert_allclose(more_embeddings[0], embeddings[0])


def test_retriever_encoder_integration_and_caching() -> None:
    """Verify RoutedDenseE5Retriever integrates with E5QueryEncoder."""
    mock_tok = MockTokenizer()
    mock_mod = MockModel(hidden_dim=4)
    encoder = E5QueryEncoder(device="cpu", model=mock_mod, tokenizer=mock_tok)

    item_ids = ["item_0", "item_1"]
    item_embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )

    retriever = RoutedDenseE5Retriever(item_ids, item_embeddings, encoder=encoder)

    # Pre-encode using retriever helper
    query_embeddings = retriever.encode_queries(["find item 0"])
    assert query_embeddings.shape == (1, 4)

    results = retriever.retrieve_global(query_embeddings, k=2)
    assert len(results) == 1
    assert len(results[0]) == 2


def test_retriever_input_validation_errors() -> None:
    """Test validation errors for invalid shapes, negative k, or dimension mismatches."""
    item_ids = ["a", "b"]
    item_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    retriever = RoutedDenseE5Retriever(item_ids, item_embeddings)

    # 1D item embeddings
    with pytest.raises(ValueError, match="must be 2D"):
        RoutedDenseE5Retriever(item_ids, np.array([1.0, 2.0]))

    # Length mismatch
    with pytest.raises(ValueError, match="length"):
        RoutedDenseE5Retriever(["a"], item_embeddings)

    # Duplicate item_ids
    with pytest.raises(ValueError, match="duplicates"):
        RoutedDenseE5Retriever(["a", "a"], item_embeddings)

    # Non-finite embeddings
    with pytest.raises(ValueError, match="finite"):
        RoutedDenseE5Retriever(
            item_ids, np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

    # Invalid k
    with pytest.raises(ValueError, match="k must be non-negative"):
        retriever.retrieve_global(np.array([[1.0, 0.0]]), k=-1)

    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension.*does not match"):
        retriever.retrieve_global(np.array([[1.0, 0.0, 0.0]]), k=1)

    # Query length mismatch in retrieve_routed
    with pytest.raises(ValueError, match="count.*does not match"):
        retriever.retrieve_routed(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            [{"a"}],  # 1 allowed set for 2 queries
            k=1,
        )

    # encode_queries without encoder
    with pytest.raises(RuntimeError, match="no encoder"):
        retriever.encode_queries(["query"])
