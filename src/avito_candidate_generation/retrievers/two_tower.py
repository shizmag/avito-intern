"""Optional torch two-tower API with deterministic text fallback for tests."""
from __future__ import annotations

from typing import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np


class TwoTowerModel:
    def __init__(self, dimension: int = 32, *, seed: int = 42) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.seed = seed

    def _encode(self, values: Sequence[str]) -> np.ndarray:
        rows = []
        for value in values:
            digest = hashlib.sha256(f"{self.seed}:{value}".encode()).digest()
            raw = np.frombuffer((digest * ((self.dimension * 4 // len(digest)) + 1))[: self.dimension * 4], dtype=np.uint32).astype(np.float32)
            vector = (raw / np.float32(2**32)) - np.float32(0.5)
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows, dtype=np.float32)

    def encode_queries(self, query_batch: Sequence[str]) -> np.ndarray:
        return self._encode(query_batch)

    def encode_items(self, item_batch: Sequence[str]) -> np.ndarray:
        return self._encode(item_batch)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"dimension": self.dimension, "seed": self.seed}) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TwoTowerModel":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(int(values["dimension"]), seed=int(values["seed"]))
