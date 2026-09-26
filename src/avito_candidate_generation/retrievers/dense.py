"""Generic dense encoder boundary and reusable local embedding artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence, Any

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
    payload = json.dumps({"texts": list(texts), "metadata": metadata}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def encode_texts(encoder: TextEncoder, texts: Sequence[str], *, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        values = np.asarray(encoder.encode(texts[start:start + batch_size], batch_size=batch_size), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(texts[start:start + batch_size]):
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


def build_embedding_artifact(encoder: TextEncoder, item_ids: Sequence[str], texts: Sequence[str], *, model: str = "local", revision: str = "unknown", batch_size: int = 32, normalize: bool = True) -> EmbeddingArtifact:
    if len(item_ids) != len(texts) or len(set(item_ids)) != len(item_ids):
        raise ValueError("item ID/text mapping must be unique and aligned")
    metadata: dict[str, Any] = {"model": model, "revision": revision, "normalize": normalize, "dtype": "float32"}
    embeddings = encode_texts(encoder, texts, batch_size=batch_size, normalize=normalize)
    metadata["shape"] = list(embeddings.shape)
    metadata["text_fingerprint"] = _fingerprint(texts, metadata)
    return EmbeddingArtifact(embeddings, tuple(str(x) for x in item_ids), metadata)


def save_embedding_artifact(artifact: EmbeddingArtifact, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", artifact.embeddings)
    (out / "item_ids.json").write_text(json.dumps(list(artifact.item_ids), ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(artifact.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def load_embedding_artifact(directory: str | Path) -> EmbeddingArtifact:
    out = Path(directory)
    embeddings = np.load(out / "embeddings.npy")
    item_ids = tuple(json.loads((out / "item_ids.json").read_text(encoding="utf-8")))
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(item_ids) or not np.isfinite(embeddings).all():
        raise ValueError("invalid embedding artifact")
    return EmbeddingArtifact(embeddings, item_ids, metadata)


def retrieve_dense(query_ids: Sequence[str], query_embeddings: np.ndarray, artifact: EmbeddingArtifact, *, k: int = 500) -> pd.DataFrame:
    return exact_top_k(query_embeddings, artifact.embeddings, list(artifact.item_ids), k=k, query_ids=list(query_ids)).assign(source="dense")
