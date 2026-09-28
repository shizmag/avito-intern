import numpy as np
import pandas as pd

from avito_candidate_generation.retrievers.dense import (
    build_embedding_artifact,
    load_embedding_artifact,
    save_embedding_artifact,
)
from avito_candidate_generation.validation import stream_exact_recall_at_k


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        return np.array([[len(text), 1.0] for text in texts], dtype=np.float32)


def test_dense_artifact_roundtrip_and_batching(tmp_path):
    artifact = build_embedding_artifact(
        FakeEncoder(), ["a", "b"], ["x", "yy"], batch_size=1
    )
    assert artifact.embeddings.shape == (2, 2)
    save_embedding_artifact(artifact, tmp_path)
    loaded = load_embedding_artifact(tmp_path)
    assert loaded.item_ids == artifact.item_ids
    assert np.allclose(loaded.embeddings, artifact.embeddings)


def test_chunked_exact_top_k_matches_full_matrix_with_stable_ties() -> None:
    from avito_candidate_generation.retrievers.exact_search import exact_top_k

    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    items = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32
    )
    ids = ["z", "b", "a", "n"]
    expected_rows = []
    for query_id, scores in zip(("q0", "q1"), queries @ items.T, strict=True):
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:3]
        expected_rows.extend(
            (query_id, ids[i], float(scores[i]), rank)
            for rank, i in enumerate(order, 1)
        )
    expected = pd.DataFrame(
        expected_rows, columns=["internal_query_id", "item_id", "score", "rank"]
    )
    actual = exact_top_k(
        queries,
        items,
        ids,
        k=3,
        query_ids=["q0", "q1"],
        batch_size=1,
        item_batch_size=2,
    )
    assert actual["item_id"].tolist() == expected["item_id"].tolist()
    assert actual["rank"].tolist() == expected["rank"].tolist()
    np.testing.assert_allclose(actual["score"], expected["score"], rtol=0.0, atol=1e-7)


def test_stream_exact_recall_uses_chunked_search() -> None:
    queries = pd.DataFrame({"internal_query_id": ["q0", "q1"], "text": ["a", "b"]})
    ground_truth = pd.DataFrame(
        {"internal_query_id": ["q0", "q1"], "item_id": ["a", "b"]}
    )
    items = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    vectors = {
        "a": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "b": np.asarray([[0.0, 1.0]], dtype=np.float32),
    }
    result = stream_exact_recall_at_k(
        queries,
        ground_truth,
        lambda texts: np.vstack([vectors[x] for x in texts]),
        items,
        ids,
        ks=(1, 2),
        query_batch_size=1,
        item_batch_size=2,
    )
    assert result == {"recall@1": 1.0, "recall@2": 1.0}
