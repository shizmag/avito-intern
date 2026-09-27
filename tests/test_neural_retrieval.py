from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch  # type: ignore[import-not-found]

from avito_candidate_generation.retrievers.dense import (
    TransformerTextEncoder,
    encode_texts,
)
from avito_candidate_generation.retrievers.two_tower import TwoTowerModel
from avito_candidate_generation.training.two_tower import (
    TrainingConfig,
    train_two_tower,
)


class _LinearEncoder:
    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


def test_encode_texts_batching_and_normalization() -> None:
    values = encode_texts(_LinearEncoder(), ["a", "bb", "ccc"], batch_size=2)
    assert values.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0)


def test_transformer_encoder_fails_without_local_model() -> None:
    with pytest.raises(RuntimeError, match="not available in local Hugging Face cache"):
        TransformerTextEncoder("definitely-not-a-local-model")


def test_two_tower_has_trainable_parameters_and_updates() -> None:
    model = TwoTowerModel(dimension=8, seed=7, vocab_size=257)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    losses = train_two_tower(
        model,
        ["красный телефон", "синий стол", "зелёная куртка"],
        ["телефон красный", "стол синий", "куртка зелёная"],
        config=TrainingConfig(epochs=2, batch_size=3, learning_rate=0.05),
    )
    assert len(losses) == 2
    assert np.isfinite(losses).all()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_two_tower_checkpoint_roundtrip(tmp_path) -> None:
    model = TwoTowerModel(dimension=8, seed=11, vocab_size=257)
    train_two_tower(
        model, ["q1", "q2"], ["i1", "i2"], config=TrainingConfig(epochs=1, batch_size=2)
    )
    path = model.save(tmp_path / "tower.pt")
    loaded = TwoTowerModel.load(path)
    np.testing.assert_allclose(
        model.encode_queries(["q1", "q2"]), loaded.encode_queries(["q1", "q2"])
    )
    np.testing.assert_allclose(
        model.encode_items(["i1", "i2"]), loaded.encode_items(["i1", "i2"])
    )


def test_query_and_item_towers_are_independent() -> None:
    model = TwoTowerModel(dimension=8, seed=3, vocab_size=257)
    assert model.query_tower is not model.item_tower
    assert not any(
        a.data_ptr() == b.data_ptr()
        for a in model.query_tower.parameters()
        for b in model.item_tower.parameters()
    )
