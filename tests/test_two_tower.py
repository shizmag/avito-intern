import numpy as np

from avito_candidate_generation.retrievers.two_tower import TwoTowerModel
from avito_candidate_generation.training.two_tower import (
    contrastive_loss,
    deterministic_batches,
    positive_mask,
)


def test_two_tower_shapes_roundtrip(tmp_path):
    model = TwoTowerModel(8, seed=3)
    assert model.encode_queries(["q"]).shape == (1, 8)
    assert np.allclose(
        model.encode_items(["i"]),
        TwoTowerModel.load(model.save(tmp_path / "model.json")).encode_items(["i"]),
    )


def test_loss_mask_and_batches():
    assert np.isfinite(contrastive_loss(np.eye(2, dtype=float)))
    assert positive_mask(["q"], ["i"], {("q", "i")}).tolist() == [[True]]
    assert [x.tolist() for x in deterministic_batches(3, 2, 42)] == [
        x.tolist() for x in deterministic_batches(3, 2, 42)
    ]
