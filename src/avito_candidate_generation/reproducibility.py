"""Deterministic random seed setup."""

from __future__ import annotations

import random
from typing import Any


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)
