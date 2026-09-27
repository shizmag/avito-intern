from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FixtureDenseEncoder:
    """Deterministic local encoder used only by smoke tests."""

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            [[float(len(text)), float(sum(ord(char) for char in text) % 97 + 1)] for text in texts],
            dtype=np.float32,
        )
