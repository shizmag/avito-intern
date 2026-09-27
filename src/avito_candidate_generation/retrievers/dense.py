"""Neural dense retrieval and validated embedding artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .exact_search import exact_top_k


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...


@dataclass(frozen=True)
class EmbeddingArtifact:
    embeddings: np.ndarray
    item_ids: tuple[str, ...]
    metadata: dict[str, Any]


def _fingerprint(texts: Sequence[str], metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {"texts": list(texts), "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_texts(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        values = np.asarray(
            encoder.encode(batch, batch_size=batch_size), dtype=np.float32
        )
        if values.ndim != 2 or values.shape[0] != len(batch):
            raise ValueError("encoder returned invalid shape")
        chunks.append(values)
    result = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("encoder returned non-finite embeddings")
    if normalize and len(result):
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero embedding cannot be normalized")
        result = result / norms
    return np.ascontiguousarray(result)


class TransformerTextEncoder:
    """Local transformer encoder; never downloads or silently falls back."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        normalize: bool = True,
        max_length: int = 256,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TransformerTextEncoder requires torch and transformers"
            ) from exc
        self._torch = torch
        self.device = torch.device(device)
        self.normalize = normalize
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path, local_files_only=True
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"model is not available in local Hugging Face cache: {model_name_or_path}"
            ) from exc
        self.model.eval()
        try:
            self.dimension = int(self.model.config.hidden_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "loaded transformer has no valid hidden dimension"
            ) from exc

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**tokens)
                mask = (
                    tokens["attention_mask"]
                    .unsqueeze(-1)
                    .to(outputs.last_hidden_state.dtype)
                )
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1)
                if self.normalize:
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(
                    pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                )
        return (
            np.vstack(rows) if rows else np.empty((0, self.dimension), dtype=np.float32)
        )


def build_embedding_artifact(
    encoder: TextEncoder,
    item_ids: Sequence[str],
    texts: Sequence[str],
    *,
    model: str = "local",
    revision: str = "unknown",
    batch_size: int = 32,
    normalize: bool = True,
) -> EmbeddingArtifact:
    if len(item_ids) != len(texts):
        raise ValueError("item_ids and texts length mismatch")
    embeddings = encode_texts(
        encoder, texts, batch_size=batch_size, normalize=normalize
    )
    try:
        dimension = (
            int(embeddings.shape[1])
            if embeddings.ndim == 2 and embeddings.shape[1] > 0
            else 0
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("encoder returned invalid embedding dimensions") from exc
    if dimension == 0 and len(texts):
        raise ValueError("encoder returned empty embedding dimension")
    metadata: dict[str, Any] = {
        "model": model,
        "revision": revision,
        "dimension": dimension,
        "normalized": normalize,
        "fingerprint": _fingerprint(texts, {"model": model, "revision": revision}),
    }
    return EmbeddingArtifact(embeddings, tuple(str(x) for x in item_ids), metadata)


def save_embedding_artifact(artifact: EmbeddingArtifact, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", artifact.embeddings)
    (out / "item_ids.json").write_text(
        json.dumps(list(artifact.item_ids), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "metadata.json").write_text(
        json.dumps(artifact.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def load_embedding_artifact(directory: str | Path) -> EmbeddingArtifact:
    out = Path(directory)
    try:
        embeddings = np.load(out / "embeddings.npy")
        item_ids = tuple(
            json.loads((out / "item_ids.json").read_text(encoding="utf-8"))
        )
        metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedding artifact: {out}") from exc
    if (
        embeddings.ndim != 2
        or len(item_ids) != embeddings.shape[0]
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("invalid embedding artifact")
    if metadata.get("dimension") != embeddings.shape[1]:
        raise ValueError("embedding metadata dimension mismatch")
    return EmbeddingArtifact(np.ascontiguousarray(embeddings), item_ids, metadata)


def retrieve_dense(
    encoder: TextEncoder,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    query_text_column: str = "text",
    item_text_column: str = "text",
    k: int = 50,
    batch_size: int = 32,
    item_artifact: EmbeddingArtifact | None = None,
) -> pd.DataFrame:
    if "item_id" not in items or "internal_query_id" not in queries:
        raise ValueError("queries/items require internal_query_id/item_id")
    if item_artifact is None:
        item_artifact = build_embedding_artifact(
            encoder,
            items["item_id"].astype(str).tolist(),
            items[item_text_column].fillna("").astype(str).tolist(),
            batch_size=batch_size,
        )
    query_embeddings = encode_texts(
        encoder,
        queries[query_text_column].fillna("").astype(str).tolist(),
        batch_size=batch_size,
    )
    return exact_top_k(
        query_embeddings,
        item_artifact.embeddings,
        list(item_artifact.item_ids),
        k=k,
        query_ids=queries["internal_query_id"].astype(str).tolist(),
    )
