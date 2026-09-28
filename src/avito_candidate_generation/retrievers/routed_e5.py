"""Routed dense retrieval using multilingual E5 embeddings with optional global fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class E5QueryEncoder:
    """Device-agnostic query encoder for E5 models with 'query: ' prefix, L2 normalization, and caching."""

    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-base",
        *,
        device: str | Any = "auto",
        max_length: int = 512,
        prefix: str = "query: ",
        normalize: bool = True,
        cache: dict[str, np.ndarray] | None = None,
        model: Any = None,
        tokenizer: Any = None,
        lazy: bool = False,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "E5QueryEncoder requires PyTorch to be installed"
            ) from exc

        self._torch = torch
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.prefix = prefix
        self.normalize = normalize
        self.local_files_only = local_files_only
        self._cache: dict[str, np.ndarray] = dict(cache) if cache is not None else {}

        self.device = self._resolve_device(device, torch)
        self.model = model
        self.tokenizer = tokenizer

        if not lazy and (self.model is None or self.tokenizer is None):
            self._ensure_loaded()

    @staticmethod
    def _resolve_device(device: str | Any, torch_mod: Any) -> Any:
        if not isinstance(device, str):
            return device
        if device == "auto":
            if torch_mod.cuda.is_available():
                return torch_mod.device("cuda")
            if (
                hasattr(torch_mod.backends, "mps")
                and torch_mod.backends.mps.is_available()
            ):
                return torch_mod.device("mps")
            return torch_mod.device("cpu")
        return torch_mod.device(device)

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required to load E5 models") from exc

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path, local_files_only=True
                )
            except (OSError, RuntimeError, ValueError):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path, local_files_only=self.local_files_only
                )

        if self.model is None:
            try:
                loaded_model = AutoModel.from_pretrained(
                    self.model_name_or_path, local_files_only=True
                )
            except (OSError, RuntimeError, ValueError):
                loaded_model = AutoModel.from_pretrained(
                    self.model_name_or_path, local_files_only=self.local_files_only
                )
            self.model = loaded_model.to(self.device)
            self.model.eval()

    def _encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        if not batch:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_loaded()
        torch = self._torch
        prefixed = [
            text if text.startswith(self.prefix) else f"{self.prefix}{text}"
            for text in batch
        ]
        with torch.inference_mode():
            raw_tokens = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if hasattr(raw_tokens, "to"):
                tokens = raw_tokens.to(self.device)
            else:
                tokens = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in raw_tokens.items()
                }
            outputs = self.model(**tokens)
            mask = (
                tokens["attention_mask"]
                .unsqueeze(-1)
                .to(outputs.last_hidden_state.dtype)
            )
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1e-9)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            result = pooled.detach().cpu().to(torch.float32).numpy()
            if (
                hasattr(self.device, "type")
                and self.device.type == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
            return result

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode queries with 'query: ' prefix, L2 normalization, and caching."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not queries:
            return np.empty((0, 0), dtype=np.float32)

        uncached = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if uncached:
            for start in range(0, len(uncached), batch_size):
                batch_texts = uncached[start : start + batch_size]
                batch_embeddings = self._encode_batch(batch_texts)
                for text, emb in zip(batch_texts, batch_embeddings, strict=True):
                    self._cache[text] = emb

        embeddings = [self._cache[q] for q in queries]
        return np.ascontiguousarray(np.vstack(embeddings), dtype=np.float32)

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """Alias for encode_queries matching TextEncoder protocol."""
        return self.encode_queries(texts, batch_size=batch_size)

    @property
    def cache(self) -> dict[str, np.ndarray]:
        """Access the underlying query embedding cache."""
        return self._cache

    def clear_cache(self) -> None:
        """Clear cached query embeddings."""
        self._cache.clear()

    def set_cached(self, query: str, embedding: np.ndarray) -> None:
        """Store a pre-encoded query embedding into cache."""
        self._cache[query] = np.asarray(embedding, dtype=np.float32)


class RoutedDenseE5Retriever:
    """Routed dense retriever using multilingual E5 item and query embeddings.

    Supports exact dot-product global retrieval, routed candidate subset retrieval,
    and optional global fallback candidate injection.
    """

    def __init__(
        self,
        item_ids: list[str],
        item_embeddings: np.ndarray,
        item_id_to_idx: dict[str, int] | None = None,
        *,
        encoder: E5QueryEncoder | None = None,
        normalize_items: bool = False,
    ) -> None:
        if item_embeddings.ndim != 2:
            raise ValueError(
                f"item_embeddings must be 2D, got shape {item_embeddings.shape}"
            )
        if len(item_ids) != item_embeddings.shape[0]:
            raise ValueError(
                f"item_ids length ({len(item_ids)}) does not match "
                f"item_embeddings rows ({item_embeddings.shape[0]})"
            )
        if not np.isfinite(item_embeddings).all():
            raise ValueError("item_embeddings must contain finite numbers")

        self.item_ids: list[str] = [str(x) for x in item_ids]
        self.item_embeddings: np.ndarray = np.ascontiguousarray(
            item_embeddings, dtype=np.float32
        )

        if normalize_items and len(self.item_embeddings) > 0:
            norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("zero embedding cannot be normalized")
            self.item_embeddings = self.item_embeddings / norms

        if item_id_to_idx is None:
            self.item_id_to_idx: dict[str, int] = {
                item_id: idx for idx, item_id in enumerate(self.item_ids)
            }
        else:
            self.item_id_to_idx = dict(item_id_to_idx)

        if len(self.item_id_to_idx) != len(self.item_ids):
            raise ValueError(
                "item_ids contains duplicates or item_id_to_idx length mismatch"
            )

        for idx in self.item_id_to_idx.values():
            if idx < 0 or idx >= len(self.item_ids):
                raise ValueError(
                    f"item_id_to_idx index {idx} out of range [0, {len(self.item_ids)})"
                )

        self.encoder = encoder

    @staticmethod
    def _top_k_candidates(
        scores: np.ndarray,
        item_ids: Sequence[str],
        k: int,
    ) -> list[tuple[str, float]]:
        n = len(scores)
        if k <= 0 or n == 0:
            return []
        if n <= k:
            candidates = np.arange(n, dtype=np.int64)
        else:
            boundary_idx = int(np.argpartition(scores, -k)[-k])
            boundary_score = float(scores[boundary_idx])
            candidates = np.flatnonzero(scores >= boundary_score)
        top_indices = sorted(
            (int(idx) for idx in candidates),
            key=lambda idx: (-float(scores[idx]), str(item_ids[idx])),
        )[:k]
        return [(str(item_ids[idx]), float(scores[idx])) for idx in top_indices]

    def retrieve_global(
        self,
        query_embeddings: np.ndarray,
        *,
        k: int,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """Exact dot product top-K over all items."""
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D, got shape {query_embeddings.shape}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if k < 0:
            raise ValueError("k must be non-negative")
        if len(query_embeddings) == 0:
            return []
        if self.item_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError(
                f"query embedding dimension ({query_embeddings.shape[1]}) does not match "
                f"item embedding dimension ({self.item_embeddings.shape[1]})"
            )
        if k == 0 or len(self.item_ids) == 0:
            return [[] for _ in range(len(query_embeddings))]

        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        results: list[list[tuple[str, float]]] = []

        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            scores_batch = batch @ self.item_embeddings.T
            for scores in scores_batch:
                results.append(self._top_k_candidates(scores, self.item_ids, k))

        return results

    def retrieve_routed(
        self,
        query_embeddings: np.ndarray,
        allowed_item_ids: Sequence[set[str]],
        *,
        k: int,
        global_fallback_k: int = 0,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """Compute dot product only against indices in allowed_item_ids to find top k.

        If global_fallback_k > 0, also retrieves top global_fallback_k from global items and appends
        any not already in the routed set. Deduplicated, sorted by score desc, item_id asc.
        """
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D, got shape {query_embeddings.shape}"
            )
        if len(query_embeddings) != len(allowed_item_ids):
            raise ValueError(
                f"query_embeddings count ({len(query_embeddings)}) does not match "
                f"allowed_item_ids count ({len(allowed_item_ids)})"
            )
        if k < 0:
            raise ValueError("k must be non-negative")
        if global_fallback_k < 0:
            raise ValueError("global_fallback_k must be non-negative")
        if len(query_embeddings) == 0:
            return []
        if self.item_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError(
                f"query embedding dimension ({query_embeddings.shape[1]}) does not match "
                f"item embedding dimension ({self.item_embeddings.shape[1]})"
            )

        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)

        global_candidates_per_query: list[list[tuple[str, float]]] | None = None
        if global_fallback_k > 0 and len(self.item_ids) > 0:
            global_candidates_per_query = self.retrieve_global(
                queries, k=global_fallback_k, batch_size=batch_size
            )

        results: list[list[tuple[str, float]]] = []

        for q_idx in range(len(queries)):
            allowed = allowed_item_ids[q_idx]
            query_vec = queries[q_idx]

            routed_candidates: list[tuple[str, float]] = []
            if k > 0 and allowed:
                valid_indices = [
                    self.item_id_to_idx[item_id]
                    for item_id in allowed
                    if item_id in self.item_id_to_idx
                ]
                if valid_indices:
                    idx_arr = np.asarray(valid_indices, dtype=np.int64)
                    sub_embeddings = self.item_embeddings[idx_arr]
                    sub_scores = sub_embeddings @ query_vec
                    sub_item_ids = [self.item_ids[idx] for idx in valid_indices]
                    routed_candidates = self._top_k_candidates(
                        sub_scores, sub_item_ids, k
                    )

            if global_fallback_k > 0 and global_candidates_per_query is not None:
                global_top = global_candidates_per_query[q_idx]
                routed_set = {item_id for item_id, _ in routed_candidates}
                fallback_candidates = [
                    (item_id, score)
                    for item_id, score in global_top
                    if item_id not in routed_set
                ]
                combined = routed_candidates + fallback_candidates
                combined.sort(key=lambda pair: (-pair[1], pair[0]))
                results.append(combined)
            else:
                results.append(routed_candidates)

        return results

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode queries using the configured encoder helper."""
        if self.encoder is None:
            raise RuntimeError(
                "retriever has no encoder configured; provide an E5QueryEncoder in __init__ "
                "or encode queries directly"
            )
        return self.encoder.encode_queries(queries, batch_size=batch_size)
