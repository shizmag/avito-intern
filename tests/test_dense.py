import numpy as np
from avito_candidate_generation.retrievers.dense import build_embedding_artifact, load_embedding_artifact, save_embedding_artifact

class FakeEncoder:
    def encode(self, texts, *, batch_size):
        return np.array([[len(text), 1.0] for text in texts], dtype=np.float32)

def test_dense_artifact_roundtrip_and_batching(tmp_path):
    artifact=build_embedding_artifact(FakeEncoder(),["a","b"],["x","yy"],batch_size=1)
    assert artifact.embeddings.shape == (2,2)
    save_embedding_artifact(artifact,tmp_path)
    loaded=load_embedding_artifact(tmp_path)
    assert loaded.item_ids == artifact.item_ids
    assert np.allclose(loaded.embeddings,artifact.embeddings)
