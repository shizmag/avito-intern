"""Contrastive two-tower training helpers with optional torch integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


def positive_mask(query_ids: Sequence[str], item_ids: Sequence[str], known_pairs: set[tuple[str, str]] | None = None) -> np.ndarray:
    pairs = known_pairs or set(zip(query_ids, item_ids))
    return np.asarray([[((q, i) in pairs) for i in item_ids] for q in query_ids], dtype=bool)


def contrastive_loss(scores: np.ndarray, positive: np.ndarray | None = None, temperature: float = 0.07) -> float:
    if scores.ndim != 2 or temperature <= 0:
        raise ValueError("scores must be 2D and temperature positive")
    labels = np.arange(scores.shape[0]) if positive is None else np.argmax(positive, axis=1)
    logits = scores / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean())


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 1
    batch_size: int = 32
    temperature: float = 0.07


def deterministic_batches(size: int, batch_size: int, seed: int = 42) -> list[np.ndarray]:
    if size < 0 or batch_size < 1:
        raise ValueError("invalid size or batch_size")
    order = np.random.default_rng(seed).permutation(size)
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]
