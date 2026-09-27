"""Trainable query/item two-tower retrieval model."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import Tensor, nn  # type: ignore[import-not-found]

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _token_id(token: str, vocab_size: int) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (vocab_size - 1) + 1


def _batch_token_ids(values: Sequence[str], vocab_size: int) -> Tensor:
    token_rows = [
        [
            _token_id(token, vocab_size)
            for token in _TOKEN_RE.findall(str(value).casefold())
        ]
        for value in values
    ]
    width = max((len(row) for row in token_rows), default=1)
    ids = torch.zeros((len(values), width), dtype=torch.long)
    for row_index, row in enumerate(token_rows):
        if row:
            ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids


class _TextTower(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dimension: int, output_dimension: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dimension, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dimension, output_dimension),
            nn.Tanh(),
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        embeddings = self.embedding(token_ids)
        mask = token_ids.ne(0).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = (embeddings * mask).sum(dim=1) / lengths
        return nn.functional.normalize(self.projection(pooled), p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Independent trainable query and item encoders.

    Text is tokenized into stable hashed token IDs only to avoid a mutable
    vocabulary artifact. Hashing selects trainable embedding rows; it does not
    produce vectors and cannot replace model training. Item encoding never sees
    query text, so item vectors can be precomputed and cached independently.
    """

    def __init__(
        self,
        dimension: int = 32,
        *,
        seed: int = 42,
        vocab_size: int = 65537,
        hidden_dimension: int | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3")
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            hidden = hidden_dimension or dimension
            self.query_tower = _TextTower(vocab_size, hidden, dimension)
            self.item_tower = _TextTower(vocab_size, hidden, dimension)
        self.dimension = dimension
        self.seed = seed
        self.vocab_size = vocab_size
        self.hidden_dimension = hidden

    def _encode_tensor(self, values: Sequence[str], tower: _TextTower) -> Tensor:
        device = next(self.parameters()).device
        token_ids = _batch_token_ids(values, self.vocab_size).to(device)
        return tower(token_ids)

    def tensor_queries(self, query_batch: Sequence[str]) -> Tensor:
        return self._encode_tensor(query_batch, self.query_tower)

    def tensor_items(self, item_batch: Sequence[str]) -> Tensor:
        return self._encode_tensor(item_batch, self.item_tower)

    def score(self, query_vectors: Tensor, item_vectors: Tensor) -> Tensor:
        if query_vectors.ndim == 2 and item_vectors.ndim == 2:
            return query_vectors @ item_vectors.T
        if query_vectors.shape == item_vectors.shape:
            return (query_vectors * item_vectors).sum(dim=-1)
        raise ValueError("query and item vectors have incompatible shapes")

    def forward(self, query_batch: Sequence[str], item_batch: Sequence[str]) -> Tensor:
        return self.score(
            self.tensor_queries(query_batch), self.tensor_items(item_batch)
        )

    def _numpy_encode(self, values: Sequence[str], tower: _TextTower) -> np.ndarray:
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            result = (
                self._encode_tensor(values, tower)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        self.train(was_training)
        return result

    def encode_queries(self, query_batch: Sequence[str]) -> np.ndarray:
        return self._numpy_encode(query_batch, self.query_tower)

    def encode_items(self, item_batch: Sequence[str]) -> np.ndarray:
        return self._numpy_encode(item_batch, self.item_tower)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file  # type: ignore[import-not-found]

            tensor_path = target.with_suffix(target.suffix + ".safetensors")
            save_file(self.state_dict(), str(tensor_path))
            metadata = {
                "schema_version": 2,
                "format": "safetensors",
                "dimension": self.dimension,
                "seed": self.seed,
                "vocab_size": self.vocab_size,
                "hidden_dimension": self.hidden_dimension,
                "tensor_file": tensor_path.name,
            }
            target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "safetensors is required for checkpoint serialization"
            ) from exc
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> TwoTowerModel:
        target = Path(path)
        try:
            metadata = json.loads(target.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != 2
                or metadata.get("format") != "safetensors"
            ):
                raise ValueError("unsupported two-tower checkpoint")
            model = cls(
                int(metadata["dimension"]),
                seed=int(metadata["seed"]),
                vocab_size=int(metadata["vocab_size"]),
                hidden_dimension=int(metadata["hidden_dimension"]),
            )
            from safetensors.torch import load_file  # type: ignore[import-not-found]

            state = load_file(
                str(target.parent / metadata["tensor_file"]),
                device=str(map_location),
            )
            model.load_state_dict(state)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(f"invalid two-tower checkpoint: {path}") from exc
        model.eval()
        return model
