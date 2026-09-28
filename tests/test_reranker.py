"""Tests for Jina reranker, pair formatting, and deterministic candidate selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from avito_candidate_generation.candidates import CandidateError
from avito_candidate_generation.config import Config
from avito_candidate_generation.provisioning import provision_model_artifact
from avito_candidate_generation.reranker import (
    JinaReranker,
    format_item_components,
    format_pair_with_priority_truncation,
    format_query_text,
    rank_reranked_candidates,
    score_candidate_pool,
)


class MockTokenizer:
    def __init__(self, char_per_token: int = 4) -> None:
        self.char_per_token = char_per_token

    def __call__(
        self,
        text_a: str | list[Any],
        text_b: str | None = None,
        *,
        truncation: bool = False,
        return_attention_mask: bool = True,
        padding: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(text_a, list):
            seqs = [f"{a} {b}" for a, b in text_a]
            max_len = (
                max(len(s) // self.char_per_token + 2 for s in seqs) if seqs else 2
            )
            if max_length is not None and truncation:
                max_len = min(max_len, max_length)
            batch_size = len(text_a)
            ids = torch_tensor([[1] * max_len for _ in range(batch_size)])
            mask = torch_tensor([[1] * max_len for _ in range(batch_size)])
            return {"input_ids": ids, "attention_mask": mask}

        doc = text_b or ""
        total_str = f"{text_a} {doc}".strip()
        n_tokens = len(total_str) // self.char_per_token + 2
        return {"input_ids": [1] * n_tokens}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        n = max(1, len(text) // self.char_per_token)
        return list(range(n))

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        chars = len(tokens) * self.char_per_token
        return "a" * chars


def torch_tensor(data: list[list[int]]) -> MagicMock:
    mock = MagicMock()
    mock.to = MagicMock(return_value=mock)
    mock.shape = (len(data), len(data[0]) if data else 0)
    return mock


class MockReranker:
    def __init__(
        self, scores: list[float] | None = None, max_length: int = 512
    ) -> None:
        self.tokenizer = MockTokenizer()
        self.max_length = max_length
        self._scores = scores or [1.0, 0.5, -0.2]

    def score_pairs(
        self, pairs: list[tuple[str, str]], *, batch_size: int = 64
    ) -> list[float]:
        return [self._scores[i % len(self._scores)] for i in range(len(pairs))]


def test_format_query_text_variations() -> None:
    assert format_query_text("iphone", "128gb") == "Запрос: iphone\nФильтры: 128gb"
    assert format_query_text("iphone", "") == "Запрос: iphone"
    assert format_query_text("iphone", "nan") == "Запрос: iphone"
    assert format_query_text("", "128gb") == "Фильтры: 128gb"
    assert format_query_text("", "") == ""
    assert format_query_text(None, None) == ""  # pyright: ignore[reportArgumentType]


def test_format_item_components_variations() -> None:
    prefix, desc = format_item_components("Телефон", "Цвет черный", "Хорошее состояние")
    assert prefix == "Название: Телефон\nПараметры: Цвет черный"
    assert desc == "Описание: Хорошее состояние"

    prefix2, desc2 = format_item_components("Стол", "", "")
    assert prefix2 == "Название: Стол"
    assert desc2 == ""

    prefix3, desc3 = format_item_components("", "Размер XL", "")
    assert prefix3 == "Параметры: Размер XL"
    assert desc3 == ""

    prefix4, desc4 = format_item_components("nan", "nan", "nan")
    assert prefix4 == ""
    assert desc4 == ""


def test_format_pair_with_priority_truncation_fits_and_truncates() -> None:
    tokenizer = MockTokenizer(char_per_token=4)
    q = "Запрос: авто"
    prefix = "Название: Лада Калина\nПараметры: 2012 г."
    desc = "Описание: " + "очень много текста " * 50

    q_out, doc_out = format_pair_with_priority_truncation(
        tokenizer, q, prefix, desc, max_length=200
    )
    assert q_out == q
    assert prefix in doc_out

    # Exceeds max_length budget:
    q_short, doc_trunc = format_pair_with_priority_truncation(
        tokenizer, q, prefix, desc, max_length=20
    )
    assert q_short == q
    assert prefix in doc_trunc


def test_jina_reranker_raises_on_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        JinaReranker(missing)


def test_score_candidate_pool_and_caching(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q1", "q2"],
            "item_id": ["i1", "i2", "i1"],
            "source": ["rrf", "rrf", "rrf"],
            "score": [0.5, 0.4, 0.9],
            "rank": [1, 2, 1],
        }
    )
    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "search_query": ["купить авто", "ноутбук"],
            "search_infm_params_text": ["", "8гб"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "item_title_raw": ["Авто Лада", "Ноутбук Asus"],
            "item_infm_params_text": ["2010", "16gb"],
            "item_description_raw": ["Без пробега", "Отличное состояние"],
        }
    )

    reranker = MockReranker(scores=[2.5, 1.2, 3.8])  # type: ignore[arg-type]
    cache_dir = tmp_path / "cache"

    scored = score_candidate_pool(
        candidates,
        queries,
        items,
        reranker,  # type: ignore[arg-type]
        pair_batch_size=2,
        query_chunk_size=1,
        cache_dir=cache_dir,
    )
    assert len(scored) == 3
    assert "jina_score" in scored.columns
    assert len(list(cache_dir.glob("*.parquet"))) == 2

    # Second run should hit cache
    scored_cached = score_candidate_pool(
        candidates,
        queries,
        items,
        reranker,  # type: ignore[arg-type]
        pair_batch_size=2,
        query_chunk_size=1,
        cache_dir=cache_dir,
    )
    assert len(scored_cached) == 3
    np.testing.assert_allclose(scored["jina_score"], scored_cached["jina_score"])


def test_score_candidate_pool_empty() -> None:
    candidates = pd.DataFrame(columns=["internal_query_id", "item_id"])
    queries = pd.DataFrame(columns=["internal_query_id"])
    items = pd.DataFrame(columns=["item_id"])
    reranker = MockReranker()
    scored = score_candidate_pool(candidates, queries, items, reranker)  # type: ignore[arg-type]
    assert scored.empty
    assert "jina_score" in scored.columns


def test_score_candidate_pool_missing_columns() -> None:
    candidates = pd.DataFrame({"invalid": [1, 2]})
    queries = pd.DataFrame()
    items = pd.DataFrame()
    reranker = MockReranker()
    with pytest.raises(CandidateError):
        score_candidate_pool(candidates, queries, items, reranker)  # type: ignore[arg-type]


def test_rank_reranked_candidates_deterministic_tie_breaking() -> None:
    scored = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q1", "q1"],
            "item_id": ["item_b", "item_a", "item_c"],
            "jina_score": [1.5, 1.5, 0.5],
        }
    )
    ranked = rank_reranked_candidates(scored, score_column="jina_score", limit=2)
    assert len(ranked) == 2
    # item_a comes before item_b because of tie-breaking on item_id asc
    assert ranked["item_id"].tolist() == ["item_a", "item_b"]
    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["source"].tolist() == ["jina_reranker", "jina_reranker"]


def test_rank_reranked_candidates_empty() -> None:
    ranked = rank_reranked_candidates(pd.DataFrame())
    assert ranked.empty
    assert list(ranked.columns) == [
        "internal_query_id",
        "item_id",
        "source",
        "score",
        "rank",
    ]


def test_provision_model_artifact_offline_and_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "mock_jina"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"dummy_weights")

    cfg = Config(path=tmp_path / "config.toml", root=tmp_path, hash="abc", values={})
    manifest = provision_model_artifact(
        "jinaai/jina-reranker-v2-base-multilingual",
        "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
        str(model_dir),
        cfg,
        allow_network=False,
        trust_remote_code=True,
        remote_code_reason="Custom XLM-RoBERTa",
    )
    assert manifest["status"] == "PASS"
    assert manifest["trust_remote_code"] is True
    assert (model_dir / "model_manifest.json").is_file()
    saved = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))
    assert saved["model"] == "jinaai/jina-reranker-v2-base-multilingual"
    assert saved["n_files"] == 3


def test_provision_model_artifact_missing_offline_raises(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty_model"
    model_dir.mkdir(parents=True)
    cfg = Config(path=tmp_path / "config.toml", root=tmp_path, hash="abc", values={})
    with pytest.raises(RuntimeError, match="model is not provisioned locally"):
        provision_model_artifact(
            "jinaai/jina-reranker-v2-base-multilingual",
            "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
            str(model_dir),
            cfg,
            allow_network=False,
        )


def test_sample_representative_queries_stratification_and_determinism() -> None:
    from avito_candidate_generation.rerank_experiment import (
        compute_metrics_suite,
        sample_representative_queries,
    )

    queries = pd.DataFrame(
        {
            "internal_query_id": [f"q{i}" for i in range(20)],
            "search_query": ["short q"] * 10
            + ["very long query with multiple words in text"] * 10,
            "search_infm_params_text": ["filters"] * 5
            + [""] * 5
            + ["filters"] * 5
            + [""] * 5,
        }
    )
    gt = pd.DataFrame(
        {
            "internal_query_id": [f"q{i}" for i in range(20)]
            + ["q0", "q1"],  # q0 and q1 multi-positive
            "item_id": [f"i{i}" for i in range(20)] + ["i99", "i98"],
        }
    )

    sampled1 = sample_representative_queries(queries, gt, sample_size=8, seed=42)
    sampled2 = sample_representative_queries(queries, gt, sample_size=8, seed=42)
    assert len(sampled1) == 8
    assert (
        sampled1["internal_query_id"].tolist() == sampled2["internal_query_id"].tolist()
    )

    # Empty queries edge case
    empty_sampled = sample_representative_queries(pd.DataFrame(), gt, sample_size=5)
    assert empty_sampled.empty

    # Metrics suite
    cand = pd.DataFrame(
        {
            "internal_query_id": ["q0", "q0", "q1"],
            "item_id": ["i0", "i99", "i1"],
            "source": ["test", "test", "test"],
            "score": [1.0, 0.9, 0.8],
            "rank": [1, 2, 1],
        }
    )
    metrics = compute_metrics_suite(cand, gt, ks=[1, 2, 5])
    assert "recall@1" in metrics
    assert "recall@2" in metrics
    assert "recall@5" in metrics


def test_safe_int_helper() -> None:
    from avito_candidate_generation.rerank_experiment import _safe_int

    assert _safe_int(10, 0) == 10
    assert _safe_int("20", 0) == 20
    assert _safe_int(None, 5) == 5
    assert _safe_int("invalid", 5) == 5


def test_run_reranker_validation_mocked(tmp_path: Path) -> None:
    from avito_candidate_generation.rerank_experiment import run_reranker_validation

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[project]
seed = 42
repo_root = "."

[artifacts]
root = "artifacts/selected"

[data]
train = "data/train.parquet"
benchmark_queries = "data/benchmark_queries.parquet"
benchmark_items = "data/benchmark_items.parquet"

[selection]
category_policy = "none"

[dense]
model_path = "artifacts/models/multilingual-e5-base"

[fusion]
rrf_k = 60

[reranker]
local_path = "mock/path"
max_length = 128
""",
        encoding="utf-8",
    )

    queries = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "search_query": ["авто", "ноутбук"],
            "search_infm_params_text": ["", "8gb"],
            "text": ["авто", "ноутбук 8gb"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "item_title_raw": ["Лада", "Asus"],
            "item_infm_params_text": ["", ""],
            "item_description_raw": ["", ""],
            "text": ["Лада", "Asus"],
        }
    )
    gt = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "item_id": ["i1", "i2"],
        }
    )

    cand1 = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "item_id": ["i1", "i2"],
            "source": ["bm25", "bm25"],
            "score": [1.0, 1.0],
            "rank": [1, 1],
        }
    )
    cand2 = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "item_id": ["i2", "i1"],
            "source": ["dense", "dense"],
            "score": [0.8, 0.8],
            "rank": [1, 1],
        }
    )
    cand3 = pd.DataFrame(
        {
            "internal_query_id": ["q1", "q2"],
            "item_id": ["i1", "i2"],
            "source": ["two_tower", "two_tower"],
            "score": [0.5, 0.5],
            "rank": [1, 1],
        }
    )

    wf_data_mock = MagicMock()
    wf_data_mock.validation_queries = queries
    wf_data_mock.validation_items = items
    wf_data_mock.validation_ground_truth = gt

    mock_emb = MagicMock()

    with (
        patch(
            "avito_candidate_generation.rerank_experiment._load_workflow_data",
            return_value=wf_data_mock,
        ),
        patch(
            "avito_candidate_generation.rerank_experiment.load_embedding_artifact",
            return_value=mock_emb,
        ),
        patch("avito_candidate_generation.rerank_experiment.TransformerTextEncoder"),
        patch("avito_candidate_generation.rerank_experiment.TwoTowerModel.load"),
        patch(
            "avito_candidate_generation.rerank_experiment._retrieve_sources",
            return_value=[cand1, cand2, cand3],
        ),
        patch(
            "avito_candidate_generation.rerank_experiment.JinaReranker"
        ) as mock_jina_cls,
    ):
        mock_reranker_inst = MagicMock()
        mock_reranker_inst.device = "cpu"
        mock_reranker_inst.tokenizer = MockTokenizer()
        mock_reranker_inst.score_pairs = MagicMock(return_value=[1.0, 0.5, 0.8, 0.2])
        mock_jina_cls.return_value = mock_reranker_inst

        out_json = tmp_path / "metrics.json"
        report = run_reranker_validation(
            cfg_file,
            sample_size=2,
            candidate_k=2,
            device="cpu",
            cache_dir=tmp_path / "cache",
            output_json=out_json,
        )

        assert report["status"] == "PASS"
        assert report["sample_size"] == 2
        assert out_json.is_file()
