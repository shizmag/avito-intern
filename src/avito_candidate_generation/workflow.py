"""Artifact-producing selected candidate-generation workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .artifacts import git_info
from .config import Config, load_config, resolve_path
from .data import build_canonical
from .evaluation import evaluate_candidates
from .fusion.rrf import reciprocal_rank_fusion
from .inference import rank_predictions
from .pipeline import FinalPipeline, SelectedManifest
from .retrievers.bm25 import BM25Index, retrieve_bm25
from .retrievers.dense import (
    EmbeddingArtifact,
    TextEncoder,
    TransformerTextEncoder,
    build_embedding_artifact,
    encode_texts,
    save_embedding_artifact,
)
from .retrievers.exact_search import exact_top_k
from .retrievers.two_tower import TwoTowerModel
from .splits import assign_item_splits, build_ground_truth
from .submission import write_submission
from .training.hard_negatives import mine_hard_negatives
from .training.two_tower import (
    TrainingConfig,
    train_two_tower,
    train_two_tower_with_hard_negatives,
)


class SmokeDenseEncoder:
    """Deterministic encoder used only by explicit local smoke command."""

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            [
                [
                    len(text),
                    sum(ord(char) for char in text) % 97 + 1,
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class WorkflowData:
    train_queries: pd.DataFrame
    train_items: pd.DataFrame
    train_pairs: pd.DataFrame
    validation_queries: pd.DataFrame
    validation_items: pd.DataFrame
    validation_ground_truth: pd.DataFrame
    benchmark_queries: pd.DataFrame
    benchmark_items: pd.DataFrame


def _frame(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("expected pandas DataFrame")
    return value


def _data_config(config: Config) -> Mapping[str, Any]:
    data = config.values.get("data")
    if not isinstance(data, dict):
        raise TypeError("selected workflow requires [data] config")
    return data


def _data_path(
    config: Config,
    data: Mapping[str, Any],
    key: str,
    aliases: tuple[str, ...] = (),
) -> Path:
    value = next((data[name] for name in (key, *aliases) if name in data), None)
    if not isinstance(value, str):
        raise TypeError(f"selected workflow requires data.{key}")
    path = resolve_path(config, value)
    if not path.is_file():
        raise FileNotFoundError(f"missing workflow input {key}: {path}")
    return path


def _compose_query_text(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "internal_query_id" not in work:
        if "query_id" not in work:
            raise ValueError("query table requires internal_query_id or query_id")
        work["internal_query_id"] = work["query_id"].astype(str)
    if "text" in work:
        result = _frame(work[["internal_query_id", "text"]].copy())
    else:
        columns = [
            column
            for column in ("search_query", "search_infm_params_text", "search_category")
            if column in work
        ]
        if not columns:
            raise ValueError("query table has no text fields")
        text = work[columns].fillna("").astype(str).agg(" ".join, axis=1)
        result = pd.DataFrame(
            {"internal_query_id": work["internal_query_id"].astype(str), "text": text}
        )
    result["internal_query_id"] = result["internal_query_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    if "query_id" in work:
        result["query_id"] = work["query_id"].astype(str).tolist()
    return _frame(result.reset_index(drop=True))


def _compose_item_text(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "item_id" not in work:
        raise ValueError("item table requires item_id")
    if "text" in work:
        result = _frame(work[["item_id", "text"]].copy())
    else:
        columns = [
            column
            for column in (
                "item_title_raw",
                "item_infm_params_text",
                "item_description_raw",
            )
            if column in work
        ]
        if not columns:
            raise ValueError("item table has no text fields")
        text = work[columns].fillna("").astype(str).agg(" ".join, axis=1)
        result = pd.DataFrame({"item_id": work["item_id"].astype(str), "text": text})
    result["item_id"] = result["item_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    return _frame(result.reset_index(drop=True))


def _normalise_queries(frame: pd.DataFrame) -> pd.DataFrame:
    result = _compose_query_text(frame)
    if bool(result["internal_query_id"].duplicated().any()):
        raise ValueError("query IDs must be unique in workflow input")
    return result


def _normalise_items(frame: pd.DataFrame) -> pd.DataFrame:
    result = _compose_item_text(frame)
    if bool(result["item_id"].duplicated().any()):
        raise ValueError("item IDs must be unique in workflow input")
    return result


def _prepare_workflow_paths(config: Config, data: Mapping[str, Any]) -> dict[str, Path]:
    keys = (
        "train_queries",
        "train_items",
        "train_pairs",
        "validation_queries",
        "validation_items",
        "validation_ground_truth",
        "benchmark_queries",
        "benchmark_items",
    )
    if all(key in data for key in keys):
        return {key: _data_path(config, data, key) for key in keys}
    raw_train = _data_path(config, data, "train")
    raw_queries = _data_path(config, data, "benchmark_queries")
    raw_items = _data_path(config, data, "benchmark_items")
    root = (
        resolve_path(
            config,
            str(config.values.get("artifacts", {}).get("root", "artifacts/selected")),
        )
        / "prepared"
    )
    canonical = build_canonical(raw_train, raw_queries, raw_items, root / "data")
    interactions = _frame(pd.read_parquet(canonical["interactions"]))
    items = _frame(pd.read_parquet(canonical["items_train"]))
    queries = _frame(pd.read_parquet(canonical["queries_train"]))
    split_frame = assign_item_splits(
        items["item_id"].astype(str).tolist(), seed=config.seed
    )
    ground_truth = build_ground_truth(interactions, split_frame)
    train_ids = (
        split_frame.loc[split_frame["split"] == "train", "item_id"].astype(str).tolist()
    )
    validation_ids = (
        split_frame.loc[split_frame["split"] == "validation", "item_id"]
        .astype(str)
        .tolist()
    )
    train_pairs = interactions[
        interactions["item_id"].astype(str).isin(train_ids)
    ].copy()
    validation_ground_truth = ground_truth.loc[
        ground_truth["split"] == "validation", ["internal_query_id", "item_id"]
    ].copy()
    train_query_ids = train_pairs["internal_query_id"].astype(str).tolist()
    validation_query_ids = (
        validation_ground_truth["internal_query_id"].astype(str).tolist()
    )
    prepared = {
        "train_queries": queries[
            queries["internal_query_id"].astype(str).isin(train_query_ids)
        ],
        "train_items": items[items["item_id"].astype(str).isin(train_ids)],
        "train_pairs": train_pairs,
        "validation_queries": queries[
            queries["internal_query_id"].astype(str).isin(validation_query_ids)
        ],
        "validation_items": items[items["item_id"].astype(str).isin(validation_ids)],
        "validation_ground_truth": validation_ground_truth,
        "benchmark_queries": _frame(pd.read_parquet(canonical["benchmark_queries"])),
        "benchmark_items": _frame(pd.read_parquet(canonical["benchmark_items"])),
    }
    paths: dict[str, Path] = {}
    for name, frame in prepared.items():
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _load_workflow_data(config: Config) -> WorkflowData:
    data = _data_config(config)
    paths = _prepare_workflow_paths(config, data)
    train_queries = _normalise_queries(_frame(pd.read_parquet(paths["train_queries"])))
    train_items = _normalise_items(_frame(pd.read_parquet(paths["train_items"])))
    train_pairs = _frame(pd.read_parquet(paths["train_pairs"])).copy()
    required_pairs = {"internal_query_id", "item_id"}
    if not required_pairs.issubset(train_pairs.columns):
        raise ValueError("train_pairs requires internal_query_id and item_id")
    train_pairs["internal_query_id"] = train_pairs["internal_query_id"].astype(str)
    train_pairs["item_id"] = train_pairs["item_id"].astype(str)
    if "query_text" not in train_pairs:
        lookup = train_queries[["internal_query_id", "text"]].copy()
        lookup["query_text"] = lookup["text"].astype(str)
        lookup = lookup[["internal_query_id", "query_text"]]
        train_pairs = _frame(
            train_pairs.merge(
                cast(pd.DataFrame, lookup),
                on="internal_query_id",
                how="left",
                validate="many_to_one",
            )
        )
    if "item_text" not in train_pairs:
        lookup = train_items[["item_id", "text"]].copy()
        lookup["item_text"] = lookup["text"].astype(str)
        lookup = lookup[["item_id", "item_text"]]
        train_pairs = _frame(
            train_pairs.merge(
                cast(pd.DataFrame, lookup),
                on="item_id",
                how="left",
                validate="many_to_one",
            )
        )
    if bool(train_pairs[["query_text", "item_text"]].isna().to_numpy().any()):
        raise ValueError("train_pairs references unknown query or item")

    validation_queries = _normalise_queries(
        _frame(pd.read_parquet(paths["validation_queries"]))
    )
    validation_items = _normalise_items(
        _frame(
            pd.read_parquet(
                _data_path(config, data, "validation_items", ("items", "train_items"))
            )
        )
    )
    ground_truth = _frame(pd.read_parquet(paths["validation_ground_truth"]))
    if "split" in ground_truth:
        ground_truth = _frame(
            ground_truth.loc[ground_truth["split"] == "validation"].copy()
        )
    validation_ground_truth = _frame(
        ground_truth[["internal_query_id", "item_id"]].drop_duplicates()
    )
    benchmark_queries = _normalise_queries(
        _frame(pd.read_parquet(paths["benchmark_queries"]))
    )
    benchmark_items = _normalise_items(
        _frame(pd.read_parquet(paths["benchmark_items"]))
    )
    return WorkflowData(
        train_queries=train_queries,
        train_items=train_items,
        train_pairs=train_pairs,
        validation_queries=validation_queries,
        validation_items=validation_items,
        validation_ground_truth=validation_ground_truth,
        benchmark_queries=benchmark_queries,
        benchmark_items=benchmark_items,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _save_embeddings(
    encoder: TextEncoder, items: pd.DataFrame, directory: Path, model: str
) -> EmbeddingArtifact:
    artifact = build_embedding_artifact(
        encoder,
        items["item_id"].tolist(),
        items["text"].tolist(),
        model=model,
        revision="local-cache-required",
        batch_size=32,
    )
    save_embedding_artifact(artifact, directory)
    return artifact


def _save_tower_embeddings(
    model: TwoTowerModel, items: pd.DataFrame, directory: Path, stage: str
) -> EmbeddingArtifact:
    embeddings = model.encode_items(items["text"].tolist())
    try:
        dimension = int(embeddings.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("invalid two-tower embedding shape") from exc
    artifact = EmbeddingArtifact(
        embeddings,
        tuple(items["item_id"].tolist()),
        {
            "model": "TwoTowerModel",
            "stage": stage,
            "dimension": dimension,
            "normalized": True,
        },
    )
    save_embedding_artifact(artifact, directory)
    return artifact


def _retrieve_sources(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    encoder: TextEncoder,
    generic: EmbeddingArtifact,
    tower: TwoTowerModel,
    k: int,
) -> list[pd.DataFrame]:
    bm25 = retrieve_bm25(queries, items, k=k)
    dense = exact_top_k(
        encode_texts(encoder, queries["text"].tolist(), batch_size=32),
        generic.embeddings,
        list(generic.item_ids),
        k=k,
        query_ids=queries["internal_query_id"].tolist(),
        source="dense",
    )
    tower_candidates = exact_top_k(
        tower.encode_queries(queries["text"].tolist()),
        tower.encode_items(items["text"].tolist()),
        items["item_id"].tolist(),
        k=k,
        query_ids=queries["internal_query_id"].tolist(),
        source="two_tower",
    )
    return [bm25, dense, tower_candidates]


def _union(sources: list[pd.DataFrame]) -> pd.DataFrame:
    work = _frame(
        pd.concat(sources, ignore_index=True).sort_values(
            ["internal_query_id", "score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    work = _frame(
        work.drop_duplicates(["internal_query_id", "item_id"], keep="first").copy()
    )
    work["source"] = "union"
    work["rank"] = work.groupby("internal_query_id").cumcount() + 1
    return _frame(
        work[["internal_query_id", "item_id", "source", "score", "rank"]].reset_index(
            drop=True
        )
    )


def _component(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def train_selected(
    config_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    encoder: TextEncoder | None = None,
) -> Path:
    config = load_config(config_path)
    data = _load_workflow_data(config)
    selection = config.values.get("selection", {})
    dense_config = config.values.get("dense", {})
    tower_config = config.values.get("two_tower", {})
    configured_root = config.values.get("artifacts", {}).get(
        "root", "artifacts/selected"
    )
    root = (
        Path(artifact_root)
        if artifact_root is not None
        else resolve_path(config, str(configured_root))
    )
    root.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        model_name = str(
            dense_config.get(
                "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        )
        try:
            max_length = int(dense_config.get("max_length", 256))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dense max_length configuration") from exc
        encoder = TransformerTextEncoder(
            model_name,
            device=str(dense_config.get("device", "cpu")),
            max_length=max_length,
        )
    else:
        model_name = str(dense_config.get("model", encoder.__class__.__name__))
    try:
        k = int(selection.get("retrieval_k", 500))
        rrf_k = int(config.values.get("fusion", {}).get("rrf_k", 60))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid selected retrieval configuration") from exc

    lexical_dir = root / "lexical"
    lexical_dir.mkdir(parents=True, exist_ok=True)
    BM25Index.fit(data.train_items).save(lexical_dir / "bm25.json")
    generic_train = _save_embeddings(
        encoder, data.train_items, root / "embeddings/generic/train", model_name
    )
    try:
        tower_config_obj = TrainingConfig(
            seed=config.seed,
            epochs=int(tower_config.get("epochs", 1)),
            batch_size=int(tower_config.get("batch_size", 32)),
            temperature=float(tower_config.get("temperature", 0.07)),
            learning_rate=float(tower_config.get("learning_rate", 1e-3)),
            device=str(tower_config.get("device", "cpu")),
        )
        tower_dimension = int(tower_config.get("dimension", 32))
        tower_vocab_size = int(tower_config.get("vocab_size", 65537))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid two-tower configuration") from exc
    tower_v1 = TwoTowerModel(
        tower_dimension, seed=config.seed, vocab_size=tower_vocab_size
    )
    losses_v1 = train_two_tower(
        tower_v1,
        data.train_pairs["query_text"].astype(str).tolist(),
        data.train_pairs["item_text"].astype(str).tolist(),
        config=tower_config_obj,
    )
    v1_path = root / "two_tower/v1/checkpoint.json"
    tower_v1.save(v1_path)
    _save_tower_embeddings(
        tower_v1, data.train_items, root / "embeddings/two_tower/v1/train", "v1"
    )

    train_sources = _retrieve_sources(
        data.train_queries, data.train_items, encoder, generic_train, tower_v1, k
    )
    positives = cast(pd.DataFrame, data.train_pairs[["internal_query_id", "item_id"]])
    try:
        max_negatives = int(
            config.values.get("hard_negatives", {}).get("max_per_query", 20)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid hard-negative configuration") from exc
    negatives = mine_hard_negatives(
        cast(pd.DataFrame, pd.concat(train_sources, ignore_index=True)),
        positives,
        max_per_query=max_negatives,
    )
    if negatives.empty:
        raise ValueError("selected workflow produced no hard negatives")
    negative_path = root / "two_tower/hard_negatives/negatives.parquet"
    negative_path.parent.mkdir(parents=True, exist_ok=True)
    negatives.to_parquet(negative_path, index=False)
    _write_json(
        root / "two_tower/hard_negatives/manifest.json",
        {"schema_version": 1, "rows": len(negatives), "status": "PASS"},
    )
    negatives_by_query = {
        str(query_id): group["item_id"].astype(str).tolist()
        for query_id, group in negatives.groupby("internal_query_id")
    }
    positive_rows = data.train_pairs[
        data.train_pairs["internal_query_id"].astype(str).isin(negatives_by_query)
    ].copy()
    item_text = dict(
        zip(
            data.train_items["item_id"].astype(str),
            data.train_items["text"].astype(str),
            strict=True,
        )
    )
    if positive_rows.empty:
        raise ValueError("hard negatives do not overlap train positives")
    tower_v2 = TwoTowerModel.load(v1_path)
    losses_v2 = train_two_tower_with_hard_negatives(
        tower_v2,
        positive_rows["query_text"].astype(str).tolist(),
        positive_rows["item_text"].astype(str).tolist(),
        [
            [item_text[item_id] for item_id in negatives_by_query[str(query_id)]]
            for query_id in positive_rows["internal_query_id"].astype(str)
        ],
        config=tower_config_obj,
    )
    v2_path = root / "two_tower/v2/checkpoint.json"
    tower_v2.save(v2_path)

    generic_validation = _save_embeddings(
        encoder,
        data.validation_items,
        root / "embeddings/generic/validation",
        model_name,
    )
    _save_tower_embeddings(
        tower_v2,
        data.validation_items,
        root / "embeddings/two_tower/v2/validation",
        "v2",
    )
    validation_sources = _retrieve_sources(
        data.validation_queries,
        data.validation_items,
        encoder,
        generic_validation,
        tower_v2,
        k,
    )
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(
        ("bm25", "dense", "two_tower"), validation_sources, strict=True
    ):
        frame.to_parquet(candidate_dir / f"{name}.parquet", index=False)
    union = _union(validation_sources)
    rrf = reciprocal_rank_fusion(
        cast(pd.DataFrame, pd.concat(validation_sources, ignore_index=True)),
        rrf_k=rrf_k,
        limit=k,
    )
    union.to_parquet(candidate_dir / "union.parquet", index=False)
    rrf.to_parquet(candidate_dir / "rrf.parquet", index=False)
    metric_payload: dict[str, Any] = {
        "schema_version": 1,
        "losses": {"v1": losses_v1, "v2": losses_v2},
        "metrics": {},
    }
    for name, frame in [
        ("bm25", validation_sources[0]),
        ("dense", validation_sources[1]),
        ("two_tower", validation_sources[2]),
        ("union", union),
        ("rrf", rrf),
    ]:
        metric_payload["metrics"][name] = evaluate_candidates(
            frame,
            data.validation_ground_truth,
            ks=(50, k),
            stage=name,
            split="validation",
        )["metrics"]
    metrics_path = _write_json(root / "metrics/validation.json", metric_payload)

    benchmark_generic = _save_embeddings(
        encoder, data.benchmark_items, root / "embeddings/generic/benchmark", model_name
    )
    benchmark_sources = _retrieve_sources(
        data.benchmark_queries,
        data.benchmark_items,
        encoder,
        benchmark_generic,
        tower_v2,
        k,
    )
    benchmark_rrf = reciprocal_rank_fusion(
        cast(pd.DataFrame, pd.concat(benchmark_sources, ignore_index=True)),
        rrf_k=rrf_k,
        limit=k,
    )
    query_map = data.benchmark_queries[["internal_query_id", "query_id"]].copy()
    scored = _frame(benchmark_rrf.merge(query_map, on="internal_query_id", how="left"))
    scored["final_score"] = scored["score"]
    predictions = rank_predictions(scored, limit=50)
    prediction_path = root / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    data_config = _data_config(config)
    benchmark_queries_path = _data_path(
        config, data_config, "benchmark_queries", ("validation_queries", "queries")
    )
    benchmark_items_path = _data_path(
        config, data_config, "benchmark_items", ("validation_items", "items")
    )
    components = {
        "dense_model": model_name,
        "bm25_index": "lexical/bm25.json",
        "two_tower_checkpoint": "two_tower/v2/checkpoint.json",
        "metrics": str(metrics_path.relative_to(root)),
        "predictions": str(prediction_path.relative_to(root)),
        "benchmark_queries": str(benchmark_queries_path),
        "benchmark_items": str(benchmark_items_path),
    }
    manifest = SelectedManifest(
        1,
        "PASS",
        ("bm25", "dense", "two_tower"),
        k,
        "rrf",
        "recall@50",
        components,
        config.hash,
        git_info(config.root).get("commit"),
    )
    return FinalPipeline(root, manifest).save_manifest(components)


def evaluate_selected(manifest_path: str | Path) -> dict[str, float]:
    path = Path(manifest_path)
    selected = FinalPipeline.load(path.parent)
    metrics_path = _component(path.parent, str(selected.manifest.components["metrics"]))
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            f"{source}/{metric}": float(value)
            for source, values in payload["metrics"].items()
            for metric, value in values.items()
        }
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid selected metrics artifact") from exc


def predict_selected(manifest_path: str | Path, *, output_csv: str | Path) -> Path:
    path = Path(manifest_path)
    selected = FinalPipeline.load(path.parent)
    components = selected.manifest.components
    predictions = _component(path.parent, str(components["predictions"]))
    queries = Path(str(components["benchmark_queries"]))
    items = Path(str(components["benchmark_items"]))
    return write_submission(predictions, queries, items, Path(output_csv))
