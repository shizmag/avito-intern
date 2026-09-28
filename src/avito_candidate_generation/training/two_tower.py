"""Contrastive two-tower training and leakage-safe training-row helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..retrievers.two_tower import TwoTowerModel


def positive_mask(
    query_ids: Sequence[str],
    item_ids: Sequence[str],
    known_pairs: set[tuple[str, str]] | None = None,
) -> np.ndarray:
    pairs = (
        known_pairs
        if known_pairs is not None
        else set(zip(query_ids, item_ids, strict=True))
    )
    return np.asarray(
        [[(q, i) in pairs for i in item_ids] for q in query_ids], dtype=bool
    )


def contrastive_loss(
    scores: np.ndarray,
    positive: np.ndarray | None = None,
    temperature: float = 0.07,
) -> float:
    if scores.ndim != 2 or scores.shape[0] == 0 or temperature <= 0:
        raise ValueError("scores must be non-empty 2D and temperature positive")
    labels = (
        np.arange(scores.shape[0]) if positive is None else np.argmax(positive, axis=1)
    )
    if len(labels) != scores.shape[0] or np.any(labels >= scores.shape[1]):
        raise ValueError("positive labels do not match score matrix")
    logits = scores / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    selected = probabilities[np.arange(len(labels)), labels]
    loss = -np.log(np.maximum(selected, 1e-12)).mean()
    try:
        return float(loss)
    except (TypeError, ValueError) as exc:
        raise ValueError("contrastive loss is not finite") from exc


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 1
    batch_size: int = 32
    temperature: float = 0.07
    learning_rate: float = 1e-3
    device: str = "cpu"


def deterministic_batches(
    size: int, batch_size: int, seed: int = 42
) -> list[np.ndarray]:
    if size < 0 or batch_size < 1:
        raise ValueError("invalid size or batch_size")
    order = np.random.default_rng(seed).permutation(size)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def train_two_tower(
    model: TwoTowerModel,
    queries: Sequence[str],
    items: Sequence[str],
    *,
    config: TrainingConfig | None = None,
) -> list[float]:
    if len(queries) != len(items) or not queries:
        raise ValueError("queries and items must be non-empty and equally sized")
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]

    if config is None:
        config = TrainingConfig()
    torch.manual_seed(config.seed)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for epoch in range(config.epochs):
        batches = list(
            deterministic_batches(len(queries), config.batch_size, config.seed + epoch)
        )
        pbar = None
        try:
            from tqdm import tqdm

            pbar = tqdm(
                batches,
                desc=f"[Two-Tower] Epoch {epoch + 1}/{config.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
        except Exception:
            pbar = None
        iterable = pbar if pbar is not None else batches
        try:
            for batch_indices in iterable:
                try:
                    indices = [int(index) for index in batch_indices]
                    q = [queries[index] for index in indices]
                    i = [items[index] for index in indices]
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("invalid training batch indices") from exc
                optimizer.zero_grad(set_to_none=True)
                logits = model(q, i) / config.temperature
                labels = torch.arange(len(q), device=logits.device)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                try:
                    loss_value = loss.detach().cpu().item()
                    losses.append(float(loss_value))
                    if pbar is not None:
                        pbar.set_postfix({"loss": f"{loss_value:.4f}"})
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("training loss is not scalar") from exc
        finally:
            if pbar is not None:
                pbar.close()
    return losses


def train_two_tower_with_hard_negatives(
    model: TwoTowerModel,
    queries: Sequence[str],
    positive_items: Sequence[str],
    negative_items: Sequence[Sequence[str]],
    *,
    config: TrainingConfig | None = None,
) -> list[float]:
    """Fine-tune towers with one positive and bounded mined negatives per query."""
    if len(queries) != len(positive_items) or len(queries) != len(negative_items):
        raise ValueError("hard-negative rows must have equal lengths")
    if any(not row for row in negative_items):
        raise ValueError("each hard-negative row must be non-empty")
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]

    if config is None:
        config = TrainingConfig()
    torch.manual_seed(config.seed)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for epoch in range(config.epochs):
        for batch_indices in deterministic_batches(
            len(queries), config.batch_size, config.seed + epoch
        ):
            try:
                indices = [int(index) for index in batch_indices]
                batch_queries = [queries[index] for index in indices]
                batch_items = [positive_items[index] for index in indices]
                batch_negatives = [list(negative_items[index]) for index in indices]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError("invalid hard-negative batch") from exc
            flat_items = batch_items + [item for row in batch_negatives for item in row]
            optimizer.zero_grad(set_to_none=True)
            query_vectors = model.tensor_queries(batch_queries)
            item_vectors = model.tensor_items(flat_items)
            positive_vectors = item_vectors[: len(batch_items)]
            positive_scores = (query_vectors * positive_vectors).sum(
                dim=1, keepdim=True
            )
            negative_vectors = item_vectors[len(batch_items) :]
            negative_scores = []
            cursor = 0
            for row in batch_negatives:
                count = len(row)
                negative_scores.append(
                    (
                        query_vectors[len(negative_scores)].unsqueeze(0)
                        @ negative_vectors[cursor : cursor + count].T
                    ).squeeze(0)
                )
                cursor += count
            width = max(len(row) for row in batch_negatives)
            try:
                padded = torch.full(
                    (len(indices), width),
                    float("-inf"),
                    device=query_vectors.device,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("unable to allocate hard-negative logits") from exc
            for row_index, values in enumerate(negative_scores):
                padded[row_index, : values.numel()] = values
            logits = torch.cat([positive_scores, padded], dim=1) / config.temperature
            labels = torch.zeros(len(indices), dtype=torch.long, device=logits.device)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            try:
                losses.append(float(loss.detach().cpu().item()))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("hard-negative loss is not scalar") from exc
    return losses


def prepare_hard_training_rows(positives, negatives):
    required = {"internal_query_id", "item_id"}
    if not required.issubset(positives.columns) or not required.issubset(
        negatives.columns
    ):
        raise ValueError("positive and negative IDs required")
    if negatives.duplicated(["internal_query_id", "item_id"]).any():
        raise ValueError("duplicate hard negative")
    positive_pairs = set(
        zip(
            positives["internal_query_id"].astype(str),
            positives["item_id"].astype(str),
            strict=True,
        )
    )
    negative_pairs = set(
        zip(
            negatives["internal_query_id"].astype(str),
            negatives["item_id"].astype(str),
            strict=True,
        )
    )
    if positive_pairs & negative_pairs:
        raise ValueError("known positive supplied as hard negative")
    result = negatives.copy()
    result["label"] = 0
    return result
