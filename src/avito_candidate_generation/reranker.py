"""Multilingual cross-encoder reranker using Jina Reranker v2."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .candidates import CandidateError, validate_candidates

logger = logging.getLogger(__name__)


def format_query_text(search_query: str, search_infm_params_text: str = "") -> str:
    """Format query and search filters for cross-encoder representation."""
    query = str(search_query or "").strip()
    params = str(search_infm_params_text or "").strip()
    if params and params.lower() == "nan":
        params = ""
    if query and params:
        return f"Запрос: {query}\nФильтры: {params}"
    if query:
        return f"Запрос: {query}"
    if params:
        return f"Фильтры: {params}"
    return ""


def format_item_components(
    title: str, params: str = "", description: str = ""
) -> tuple[str, str]:
    """Format item into priority prefix (title + params) and variable description."""
    t = str(title or "").strip()
    p = str(params or "").strip()
    d = str(description or "").strip()
    if t.lower() == "nan":
        t = ""
    if p.lower() == "nan":
        p = ""
    if d.lower() == "nan":
        d = ""

    prefix_parts: list[str] = []
    if t:
        prefix_parts.append(f"Название: {t}")
    if p:
        prefix_parts.append(f"Параметры: {p}")
    prefix = "\n".join(prefix_parts)
    desc = f"Описание: {d}" if d else ""
    return prefix, desc


def format_pair_with_priority_truncation(
    tokenizer: Any,
    query_text: str,
    prefix_text: str,
    desc_text: str,
    max_length: int = 512,
) -> tuple[str, str]:
    """Format query-document pair with priority truncation preserving title and params."""
    full_doc = (
        f"{prefix_text}\n{desc_text}".strip()
        if prefix_text and desc_text
        else (prefix_text or desc_text)
    )
    if not full_doc:
        return query_text, ""

    try:
        encoded = tokenizer(
            query_text, full_doc, truncation=False, return_attention_mask=False
        )
        if len(encoded["input_ids"]) <= max_length:
            return query_text, full_doc
    except Exception as exc:
        logger.debug("Tokenizer full pair length check failed: %s", exc)

    if not desc_text or not prefix_text:
        return query_text, full_doc

    try:
        prefix_encoded = tokenizer(
            query_text, prefix_text, truncation=False, return_attention_mask=False
        )
        prefix_len = len(prefix_encoded["input_ids"])
        budget = max(0, max_length - prefix_len - 4)
        if budget <= 0:
            return query_text, prefix_text
        desc_tokens = tokenizer.encode(desc_text, add_special_tokens=False)
        truncated_tokens = desc_tokens[:budget]
        truncated_desc = tokenizer.decode(
            truncated_tokens, skip_special_tokens=True
        ).strip()
        doc = (
            f"{prefix_text}\n{truncated_desc}".strip()
            if truncated_desc
            else prefix_text
        )
        return query_text, doc
    except Exception as exc:
        logger.debug("Tokenizer priority truncation fallback: %s", exc)
        return query_text, full_doc


@dataclass(frozen=True)
class RerankerConfig:
    model_path: str
    model_id: str = "jinaai/jina-reranker-v2-base-multilingual"
    revision: str = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
    max_length: int = 512
    device: str = "auto"
    batch_size: int = 64
    candidate_k: int = 500
    final_k: int = 50


class JinaReranker:
    """Offline cross-encoder reranker backed by Jina Reranker v2."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int = 512,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self.max_length = max_length

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), trust_remote_code=True, dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 64
    ) -> list[float]:
        """Score (query, doc) text pairs in batches using no_grad/inference_mode."""
        if not pairs:
            return []
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                batch_scores = logits.detach().cpu().to(torch.float32).tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        return scores


def score_candidate_pool(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    reranker: JinaReranker,
    *,
    pair_batch_size: int = 64,
    query_chunk_size: int = 100,
    cache_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> pd.DataFrame:
    """Score candidates with memory-bounded chunking and resume support."""
    if candidates.empty:
        return candidates.assign(jina_score=pd.Series(dtype="float32"))

    required_cand = {"internal_query_id", "item_id"}
    if not required_cand.issubset(candidates.columns):
        raise CandidateError("candidates must contain internal_query_id and item_id")

    query_lookup: dict[str, str] = {}
    for _, row in queries.iterrows():
        qid = str(row["internal_query_id"])
        q_text = format_query_text(
            str(row.get("search_query", "")),
            str(row.get("search_infm_params_text", "")),
        )
        query_lookup[qid] = q_text

    item_lookup: dict[str, tuple[str, str]] = {}
    for _, row in items.iterrows():
        iid = str(row["item_id"])
        prefix, desc = format_item_components(
            str(row.get("item_title_raw", "")),
            str(row.get("item_infm_params_text", "")),
            str(row.get("item_description_raw", "")),
        )
        item_lookup[iid] = (prefix, desc)

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

    unique_query_ids = list(
        dict.fromkeys(candidates["internal_query_id"].astype(str).tolist())
    )
    total_queries = len(unique_query_ids)
    total_pairs = len(candidates)
    processed_queries = 0
    processed_pairs = 0

    chunk_frames: list[pd.DataFrame] = []

    for chunk_idx, start_q in enumerate(range(0, total_queries, query_chunk_size)):
        chunk_qids = unique_query_ids[start_q : start_q + query_chunk_size]
        chunk_file = (
            cache_path / f"part_{chunk_idx:05d}.parquet" if cache_path else None
        )

        if chunk_file is not None and chunk_file.is_file():
            try:
                cached_chunk = pd.read_parquet(chunk_file)
                if set(cached_chunk["internal_query_id"].astype(str).unique()) == set(
                    chunk_qids
                ):
                    chunk_frames.append(cast(pd.DataFrame, cached_chunk))
                    processed_queries += len(chunk_qids)
                    processed_pairs += len(cached_chunk)
                    if progress_callback:
                        progress_callback(
                            processed_queries,
                            total_queries,
                            processed_pairs,
                            total_pairs,
                        )
                    continue
            except Exception as exc:
                logger.debug("Failed reading cache chunk: %s", exc)

        chunk_cand = cast(
            pd.DataFrame,
            candidates[
                candidates["internal_query_id"].astype(str).isin(chunk_qids)
            ].copy(),
        )
        if chunk_cand.empty:
            continue

        pairs_to_score: list[tuple[str, str]] = []
        for qid, iid in zip(
            chunk_cand["internal_query_id"].astype(str),
            chunk_cand["item_id"].astype(str),
            strict=True,
        ):
            q_text = query_lookup.get(qid, "")
            prefix, desc = item_lookup.get(iid, ("", ""))
            formatted_q, formatted_doc = format_pair_with_priority_truncation(
                reranker.tokenizer,
                q_text,
                prefix,
                desc,
                max_length=reranker.max_length,
            )
            pairs_to_score.append((formatted_q, formatted_doc))

        scores = reranker.score_pairs(pairs_to_score, batch_size=pair_batch_size)
        chunk_cand["jina_score"] = np.asarray(scores, dtype=np.float32)

        if chunk_file is not None:
            chunk_cand.to_parquet(chunk_file, index=False)

        chunk_frames.append(chunk_cand)
        processed_queries += len(chunk_qids)
        processed_pairs += len(chunk_cand)
        if progress_callback:
            progress_callback(
                processed_queries, total_queries, processed_pairs, total_pairs
            )

    result = (
        pd.concat(chunk_frames, ignore_index=True)
        if chunk_frames
        else candidates.assign(jina_score=pd.Series(dtype="float32"))
    )
    return result


def rank_reranked_candidates(
    scored_candidates: pd.DataFrame,
    *,
    score_column: str = "jina_score",
    limit: int = 50,
    source_name: str = "jina_reranker",
) -> pd.DataFrame:
    """Sort candidates by score descending with stable tie-breaking and rank up to limit."""
    if scored_candidates.empty:
        return pd.DataFrame(
            columns=["internal_query_id", "item_id", "source", "score", "rank"]
        )

    required = {"internal_query_id", "item_id", score_column}
    if not required.issubset(scored_candidates.columns):
        raise CandidateError(f"missing required columns: {required}")

    work = scored_candidates.copy()
    work = work.sort_values(
        ["internal_query_id", score_column, "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work = work.drop_duplicates(["internal_query_id", "item_id"])
    ranked = work.groupby("internal_query_id", group_keys=False).head(limit).copy()
    ranked["rank"] = ranked.groupby("internal_query_id").cumcount() + 1
    ranked["source"] = source_name
    ranked["score"] = ranked[score_column].astype(np.float32)

    validate_candidates(ranked)
    return ranked.reset_index(drop=True)
