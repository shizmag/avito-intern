"""Artifact-producing selected candidate-generation workflow."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
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
from .fusion.catboost import save_selector, score_selector, train_selector
from .fusion.features import build_pair_features
from .inference import rank_predictions
from .pipeline import FinalPipeline, SelectedManifest
from .retrievers.bm25 import BM25Index, retrieve_bm25
from .retrievers.dense import (
    EmbeddingArtifact,
    TextEncoder,
    TransformerTextEncoder,
    build_embedding_artifact_streaming,
    encode_texts,
    load_cached_embedding_artifact,
    load_embedding_artifact,
    save_embedding_artifact,
)
from .retrievers.exact_search import exact_top_k
from .retrievers.two_tower import TwoTowerModel
from .splits import assign_item_splits, build_ground_truth
from .submission import write_submission
from .training.two_tower import (
    TrainingConfig,
    train_two_tower,
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
        keep = ["internal_query_id", "text"]
        if "search_category" in work:
            keep.append("search_category")
        result = _frame(work[keep].copy())
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
        if "search_category" in work:
            result["search_category"] = work["search_category"].tolist()
    result["internal_query_id"] = result["internal_query_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    if "search_category" in result:
        result["search_category"] = result["search_category"].fillna("").astype(str)
    if "query_id" in work:
        result["query_id"] = work["query_id"].astype(str).tolist()
    return _frame(result.reset_index(drop=True))


def _compose_item_text(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "item_id" not in work:
        raise ValueError("item table requires item_id")
    if "text" in work:
        keep = ["item_id", "text"]
        for metadata in ("search_category", "item_category_id"):
            if metadata in work:
                keep.append(metadata)
        result = _frame(work[keep].copy())
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
        if "search_category" in work:
            result["search_category"] = work["search_category"].tolist()
        if "item_category_id" in work:
            result["item_category_id"] = work["item_category_id"].tolist()
    result["item_id"] = result["item_id"].astype(str)
    result["text"] = result["text"].fillna("").astype(str)
    if "search_category" in result:
        result["search_category"] = result["search_category"].fillna("").astype(str)
    if "item_category_id" in result:
        result["item_category_id"] = result["item_category_id"].fillna("").astype(str)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_cache_hit(
    root: Path, inputs: Mapping[str, Path], *, seed: int
) -> dict[str, Path] | None:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("seed") != seed:
            return None
        expected_inputs = {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in inputs.items()
        }
        if manifest.get("inputs") != expected_inputs:
            return None
        output_names = manifest.get("outputs")
        if not isinstance(output_names, dict):
            return None
        paths = {name: root / str(value) for name, value in output_names.items()}
        required = {
            "train_queries",
            "train_items",
            "train_pairs",
            "validation_queries",
            "validation_items",
            "validation_ground_truth",
            "benchmark_queries",
            "benchmark_items",
        }
        if set(paths) != required or not all(path.is_file() for path in paths.values()):
            return None
        if not (root / "data" / "data_report.json").is_file():
            return None
        return paths
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_prepared_manifest(
    root: Path, inputs: Mapping[str, Path], outputs: Mapping[str, Path], seed: int
) -> None:
    payload = {
        "schema_version": 1,
        "algorithm": "workflow-prepared-v1",
        "seed": seed,
        "inputs": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            name: str(path.relative_to(root)) for name, path in outputs.items()
        },
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, root / "manifest.json")


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
    root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "train": raw_train,
        "benchmark_queries": raw_queries,
        "benchmark_items": raw_items,
    }
    cached = _prepared_cache_hit(root, inputs, seed=config.seed)
    if cached is not None:
        return cached
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
    _write_prepared_manifest(root, inputs, paths, config.seed)
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
        _frame(pd.read_parquet(paths["validation_items"]))
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
    encoder: TextEncoder,
    items: pd.DataFrame,
    directory: Path,
    model: str,
    *,
    revision: str = "unknown",
    batch_size: int = 32,
    text_prefix: str = "",
) -> EmbeddingArtifact:
    item_ids = items["item_id"].tolist()
    texts = [text_prefix + str(text) for text in items["text"].tolist()]
    cached = load_cached_embedding_artifact(
        directory,
        item_ids=item_ids,
        texts=texts,
        model=model,
        revision=revision,
    )
    if cached is not None:
        return cached
    artifact = build_embedding_artifact_streaming(
        encoder,
        item_ids,
        texts,
        directory,
        model=model,
        revision=revision,
        batch_size=batch_size,
    )
    save_embedding_artifact(artifact, directory)
    return artifact


def _save_tower_embeddings(
    model: TwoTowerModel,
    items: pd.DataFrame,
    directory: Path,
    stage: str,
    *,
    batch_size: int = 256,
) -> EmbeddingArtifact:
    directory.mkdir(parents=True, exist_ok=True)
    embedding_path = directory / "embeddings.npy"
    item_ids_path = directory / "item_ids.json"
    metadata_path = directory / "metadata.json"
    if embedding_path.is_file() and item_ids_path.is_file() and metadata_path.is_file():
        try:
            cached = load_embedding_artifact(directory)
            if len(cached.item_ids) == len(items):
                return cached
        except ValueError:
            pass
    embeddings = model.encode_items_streaming(
        items["text"].tolist(), embedding_path, batch_size=batch_size
    )
    artifact = EmbeddingArtifact(
        embeddings,
        tuple(items["item_id"].tolist()),
        {
            "model": "TwoTowerModel",
            "stage": stage,
            "dimension": model.dimension,
            "normalized": True,
        },
    )
    (directory / "item_ids.json").write_text(
        json.dumps(list(artifact.item_ids), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text(
        json.dumps(artifact.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact


def _retrieve_sources(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    encoder: TextEncoder,
    generic: EmbeddingArtifact,
    tower: TwoTowerModel,
    tower_items: EmbeddingArtifact,
    k: int,
    *,
    encoder_batch_size: int = 32,
    category_policy: str = "none",
    dense_query_prefix: str = "",
) -> list[pd.DataFrame]:
    print(
        f"  -> [BM25] Retrieving top-{k} candidates for {len(queries):,} queries...",
        flush=True,
    )
    t0 = time.time()
    bm25 = retrieve_bm25(
        queries,
        items,
        k=k,
        category_policy=category_policy,
        item_category_column="item_category_id",
    )
    print(f"     ✓ BM25 done in {time.time() - t0:.1f}s", flush=True)

    print(
        f"  -> [Dense E5] Encoding queries & exact top-{k} search...",
        flush=True,
    )
    t0 = time.time()
    dense = exact_top_k(
        encode_texts(
            encoder,
            [dense_query_prefix + str(text) for text in queries["text"].tolist()],
            batch_size=encoder_batch_size,
        ),
        generic.embeddings,
        list(generic.item_ids),
        k=k,
        query_ids=queries["internal_query_id"].tolist(),
        source="dense",
    )
    print(f"     ✓ Dense E5 search done in {time.time() - t0:.1f}s", flush=True)

    print(
        f"  -> [Two-Tower] Encoding queries & exact top-{k} search...",
        flush=True,
    )
    t0 = time.time()
    tower_candidates = exact_top_k(
        tower.encode_queries(queries["text"].tolist(), batch_size=encoder_batch_size),
        tower_items.embeddings,
        list(tower_items.item_ids),
        k=k,
        query_ids=queries["internal_query_id"].tolist(),
        source="two_tower",
    )
    print(f"     ✓ Two-Tower search done in {time.time() - t0:.1f}s", flush=True)
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


def prepare_selector_features(
    candidates: pd.DataFrame,
    *,
    source_names: tuple[str, ...] = ("bm25", "dense", "two_tower"),
    missing_rank: int,
) -> pd.DataFrame:
    """Build numeric pair features with deterministic missing-value semantics."""
    features = build_pair_features(candidates, source_names=list(source_names))
    for source in source_names:
        rank = f"{source}_rank"
        score = f"{source}_score"
        present = f"{source}_present"
        features[present] = features[present].fillna(False).astype("int8")
        features[rank] = features[rank].fillna(missing_rank).astype("int32")
        score_values = cast(list[float], features[score].dropna().tolist())
        floor = min(score_values) - 1.0 if score_values else -1.0
        features[score] = features[score].fillna(floor).astype("float32")
    features["retriever_count"] = features["retriever_count"].astype("int8")
    return features


def rank_with_selector(
    candidates: pd.DataFrame,
    model: dict[str, object],
    *,
    limit: int,
    missing_rank: int,
    source_names: tuple[str, ...] = ("bm25", "dense", "two_tower"),
) -> pd.DataFrame:
    """Score unique candidate pairs and return deterministic per-query top-K."""
    features = prepare_selector_features(
        candidates, source_names=source_names, missing_rank=missing_rank
    )
    features["score"] = score_selector(features, model)
    features = features.sort_values(
        ["internal_query_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    result = features.groupby("internal_query_id", group_keys=False).head(limit).copy()
    result["rank"] = result.groupby("internal_query_id").cumcount() + 1
    result["source"] = "catboost"
    return pd.DataFrame(
        result[["internal_query_id", "item_id", "source", "score", "rank"]]
    ).reset_index(drop=True)


def _union_chunked(sources: list[pd.DataFrame], chunk_size: int = 5000) -> pd.DataFrame:
    all_query_ids = list(
        dict.fromkeys(sources[0]["internal_query_id"].astype(str).tolist())
    )
    if len(all_query_ids) <= chunk_size:
        return _union(sources)

    parts: list[pd.DataFrame] = []
    for start in range(0, len(all_query_ids), chunk_size):
        chunk_qids = set(all_query_ids[start : start + chunk_size])
        chunk_sources = [
            src[src["internal_query_id"].astype(str).isin(chunk_qids)]
            for src in sources
        ]
        parts.append(_union(chunk_sources))
        del chunk_sources
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def rank_with_selector_chunked(
    candidates: pd.DataFrame,
    model: dict[str, object],
    *,
    limit: int,
    missing_rank: int,
    source_names: tuple[str, ...] = ("bm25", "dense", "two_tower"),
    chunk_size: int = 5000,
) -> pd.DataFrame:
    all_query_ids = list(
        dict.fromkeys(candidates["internal_query_id"].astype(str).tolist())
    )
    if len(all_query_ids) <= chunk_size:
        return rank_with_selector(
            candidates,
            model,
            limit=limit,
            missing_rank=missing_rank,
            source_names=source_names,
        )
    parts: list[pd.DataFrame] = []
    for start in range(0, len(all_query_ids), chunk_size):
        chunk_qids = set(all_query_ids[start : start + chunk_size])
        chunk_cand = candidates[
            candidates["internal_query_id"].astype(str).isin(chunk_qids)
        ].copy()
        parts.append(
            rank_with_selector(
                chunk_cand,
                model,
                limit=limit,
                missing_rank=missing_rank,
                source_names=source_names,
            )
        )
        del chunk_cand
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def _component(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def train_selected(
    config_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    encoder: TextEncoder | None = None,
    resume: bool = True,
) -> Path:
    config = load_config(config_path)
    data = _load_workflow_data(config)
    selection = config.values.get("selection", {})
    dense_config = config.values.get("dense", {})
    tower_config = config.values.get("two_tower", {})
    catboost_config = config.values.get("catboost", {})
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
        configured_model_path = Path(
            str(dense_config.get("model_path", model_name))
        ).expanduser()
        model_path = (
            configured_model_path
            if configured_model_path.is_absolute()
            else resolve_path(config, configured_model_path)
        )
        encoder = TransformerTextEncoder(
            str(model_path),
            device=str(dense_config.get("device", "auto")),
            max_length=max_length,
            revision=str(dense_config.get("revision", "unknown")),
        )
    else:
        model_name = str(dense_config.get("model", encoder.__class__.__name__))
    dense_query_prefix = str(dense_config.get("query_prefix", ""))
    dense_item_prefix = str(dense_config.get("item_prefix", ""))
    try:
        k = int(selection.get("retrieval_k", 500))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid selected retrieval configuration") from exc

    t_start = time.time()
    print("=" * 70, flush=True)
    print("Avito Candidate Generation: Training Champion Pipeline", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Artifact root: {root}", flush=True)
    print("=" * 70, flush=True)

    # [1/8] BM25 Index
    lexical_dir = root / "lexical"
    lexical_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = lexical_dir / "bm25.npz"
    if resume and bm25_path.exists() and (bm25_path / "metadata.json").is_file():
        print(
            f"\n[1/8] Reusing existing BM25 index from {bm25_path}",
            flush=True,
        )
    else:
        print(
            f"\n[1/8] Fitting BM25 index on {len(data.train_items):,} train items...",
            flush=True,
        )
        t0 = time.time()
        BM25Index.fit(data.train_items).save(bm25_path)
        print(
            f"  ✓ BM25 index saved to {bm25_path} [{time.time() - t0:.1f}s]",
            flush=True,
        )

    try:
        encoder_batch_size = int(dense_config.get("batch_size", 32))
        tower_batch_size = int(tower_config.get("encode_batch_size", 256))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid encoder batch-size configuration") from exc
    model_revision = str(dense_config.get("revision", "unknown"))

    # [2/8] Dense E5 train items
    train_dense_dir = root / "embeddings/generic/train"
    if (
        resume
        and (train_dense_dir / "embeddings.npy").is_file()
        and (train_dense_dir / "item_ids.json").is_file()
    ):
        print(
            f"\n[2/8] Reusing existing train item embeddings from {train_dense_dir}",
            flush=True,
        )
    else:
        print(
            f"\n[2/8] Encoding {len(data.train_items):,} train items with Dense E5...",
            flush=True,
        )
        t0 = time.time()
        _ = _save_embeddings(
            encoder,
            data.train_items,
            train_dense_dir,
            model_name,
            revision=model_revision,
            batch_size=encoder_batch_size,
            text_prefix=dense_item_prefix,
        )
        print(f"  ✓ Train items encoded [{time.time() - t0:.1f}s]", flush=True)

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

    # [3/8] Two-Tower v1
    v1_path = root / "two_tower/v1/checkpoint.json"
    tt_train_dir = root / "embeddings/two_tower/v1/train"
    losses_v1: list[float] = []
    if resume and v1_path.is_file() and (tt_train_dir / "embeddings.npy").is_file():
        print(
            f"\n[3/8] Reusing existing Two-Tower v1 checkpoint from {v1_path}",
            flush=True,
        )
        tower_v1 = TwoTowerModel.load(v1_path)
    else:
        print(
            f"\n[3/8] Training Two-Tower v1 on {len(data.train_pairs):,} interaction pairs...",
            flush=True,
        )
        t0 = time.time()
        tower_v1 = TwoTowerModel(
            tower_dimension, seed=config.seed, vocab_size=tower_vocab_size
        )
        losses_v1 = train_two_tower(
            tower_v1,
            data.train_pairs["query_text"].astype(str).tolist(),
            data.train_pairs["item_text"].astype(str).tolist(),
            config=tower_config_obj,
        )
        tower_v1.save(v1_path)
        _ = _save_tower_embeddings(
            tower_v1,
            data.train_items,
            tt_train_dir,
            "v1",
            batch_size=tower_batch_size,
        )
        print(
            f"  ✓ Two-Tower v1 checkpoint saved [{time.time() - t0:.1f}s]", flush=True
        )

    # [4/8] Validation items
    val_dense_dir = root / "embeddings/generic/validation"
    val_tt_dir = root / "embeddings/two_tower/v1/validation"
    selected_tower = tower_v1
    try:
        catboost_iterations = int(catboost_config.get("iterations", 250))
        catboost_depth = int(catboost_config.get("depth", 7))
        catboost_learning_rate = float(catboost_config.get("learning_rate", 0.08))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid CatBoost configuration") from exc

    if (
        resume
        and (val_dense_dir / "embeddings.npy").is_file()
        and (val_tt_dir / "embeddings.npy").is_file()
    ):
        print(
            "\n[4/8] Reusing existing validation item embeddings",
            flush=True,
        )
        generic_validation = load_embedding_artifact(val_dense_dir)
        tower_v1_validation = load_embedding_artifact(val_tt_dir)
    else:
        print(
            f"\n[4/8] Encoding {len(data.validation_items):,} validation items...",
            flush=True,
        )
        t0 = time.time()
        generic_validation = _save_embeddings(
            encoder,
            data.validation_items,
            val_dense_dir,
            model_name,
            revision=model_revision,
            batch_size=encoder_batch_size,
            text_prefix=dense_item_prefix,
        )
        tower_v1_validation = _save_tower_embeddings(
            selected_tower,
            data.validation_items,
            val_tt_dir,
            "v1",
            batch_size=tower_batch_size,
        )
        print(f"  ✓ Validation items encoded [{time.time() - t0:.1f}s]", flush=True)

    # [5/8] Validation candidates
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_files = [
        candidate_dir / f"{name}.parquet" for name in ("bm25", "dense", "two_tower")
    ]
    if resume and all(f.is_file() for f in candidate_files):
        print(
            f"\n[5/8] Reusing existing validation candidates from {candidate_dir}",
            flush=True,
        )
        validation_sources = [pd.read_parquet(f) for f in candidate_files]
    else:
        print(
            f"\n[5/8] Retrieving validation candidates for {len(data.validation_queries):,} queries (k={k})...",
            flush=True,
        )
        t0 = time.time()
        validation_sources = _retrieve_sources(
            data.validation_queries,
            data.validation_items,
            encoder,
            generic_validation,
            selected_tower,
            tower_v1_validation,
            k,
            encoder_batch_size=encoder_batch_size,
            category_policy=str(selection.get("category_policy", "none")),
            dense_query_prefix=dense_query_prefix,
        )
        print(
            f"  ✓ Validation candidates retrieved [{time.time() - t0:.1f}s]", flush=True
        )
        for name, frame in zip(
            ("bm25", "dense", "two_tower"), validation_sources, strict=True
        ):
            frame.to_parquet(candidate_dir / f"{name}.parquet", index=False)

    # [6/8] Training CatBoost selector
    tuning_query_ids: set[str] = set()
    for query_id in data.validation_queries["internal_query_id"].astype(str):
        try:
            bucket = (
                int.from_bytes(
                    hashlib.sha256(f"catboost:{query_id}".encode()).digest()[:8], "big"
                )
                % 5
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid validation query ID") from exc
        if bucket == 0:
            tuning_query_ids.add(query_id)

    validation_pairs = set(
        zip(
            data.validation_ground_truth["internal_query_id"].astype(str),
            data.validation_ground_truth["item_id"].astype(str),
            strict=True,
        )
    )

    if tuning_query_ids:
        tuning_sources = [
            src[src["internal_query_id"].astype(str).isin(tuning_query_ids)].copy()
            for src in validation_sources
        ]
        tuning_candidates = pd.concat(tuning_sources, ignore_index=True)
        del tuning_sources
        gc.collect()

        tuning_features = prepare_selector_features(
            tuning_candidates,
            missing_rank=k + 1,
        )
        del tuning_candidates
        gc.collect()

        tuning_features["label"] = [
            (str(query_id), str(item_id)) in validation_pairs
            for query_id, item_id in zip(
                tuning_features["internal_query_id"],
                tuning_features["item_id"],
                strict=True,
            )
        ]
        tuning_labels = set(tuning_features["label"].tolist())
    else:
        tuning_labels = set()

    if (
        not tuning_query_ids
        or len(tuning_features.index) == 0
        or len(tuning_labels) < 2
    ):
        all_val_candidates = pd.concat(validation_sources, ignore_index=True)
        tuning_features = prepare_selector_features(
            all_val_candidates, missing_rank=k + 1
        )
        tuning_features["label"] = [
            (str(query_id), str(item_id)) in validation_pairs
            for query_id, item_id in zip(
                tuning_features["internal_query_id"],
                tuning_features["item_id"],
                strict=True,
            )
        ]
        tuning_query_ids = set()

    print(
        f"\n[6/8] Training CatBoost selector on {len(tuning_features):,} candidate pairs...",
        flush=True,
    )
    t0 = time.time()
    selector = train_selector(
        tuning_features,
        iterations=catboost_iterations,
        depth=catboost_depth,
        learning_rate=catboost_learning_rate,
        random_seed=config.seed,
    )
    selector_path = save_selector(selector, root / "fusion/catboost_selector.json")
    del tuning_features
    gc.collect()

    heldout_sources = [
        src[~src["internal_query_id"].astype(str).isin(tuning_query_ids)].copy()
        for src in validation_sources
    ]
    heldout_candidates = pd.concat(heldout_sources, ignore_index=True)
    del heldout_sources
    gc.collect()

    catboost_ranked = rank_with_selector_chunked(
        heldout_candidates,
        selector,
        limit=50,
        missing_rank=k + 1,
    )
    del heldout_candidates
    gc.collect()

    union = _union_chunked(validation_sources)
    union.to_parquet(candidate_dir / "union.parquet", index=False)
    catboost_ranked.to_parquet(candidate_dir / "catboost.parquet", index=False)

    metric_payload: dict[str, Any] = {
        "schema_version": 1,
        "losses": {"v1": losses_v1},
        "metrics": {},
    }
    for name, frame in [
        ("bm25", validation_sources[0]),
        ("dense", validation_sources[1]),
        ("two_tower", validation_sources[2]),
        ("union", union),
    ]:
        metric_payload["metrics"][name] = evaluate_candidates(
            frame,
            data.validation_ground_truth,
            ks=(50, k),
            stage=name,
            split="validation",
        )["metrics"]
    heldout_ground_truth = cast(
        pd.DataFrame,
        data.validation_ground_truth.loc[
            ~data.validation_ground_truth["internal_query_id"]
            .astype(str)
            .isin(list(tuning_query_ids))
        ].copy(),
    )
    metric_payload["metrics"]["catboost"] = evaluate_candidates(
        catboost_ranked,
        heldout_ground_truth,
        ks=(50, k),
        stage="catboost",
        split="validation-heldout",
    )["metrics"]
    metrics_path = _write_json(root / "metrics/validation.json", metric_payload)
    print(
        f"  ✓ CatBoost selector trained and validation metrics written [{time.time() - t0:.1f}s]",
        flush=True,
    )

    # [7/8] Benchmark items
    print(
        f"\n[7/8] Encoding {len(data.benchmark_items):,} benchmark items...",
        flush=True,
    )
    t0 = time.time()
    benchmark_generic = _save_embeddings(
        encoder,
        data.benchmark_items,
        root / "embeddings/generic/benchmark",
        model_name,
        revision=model_revision,
        batch_size=encoder_batch_size,
        text_prefix=dense_item_prefix,
    )
    tower_v1_benchmark = _save_tower_embeddings(
        selected_tower,
        data.benchmark_items,
        root / "embeddings/two_tower/v1/benchmark",
        "v1",
        batch_size=tower_batch_size,
    )
    print(f"  ✓ Benchmark items encoded [{time.time() - t0:.1f}s]", flush=True)

    # [8/8] Benchmark predictions
    print(
        f"\n[8/8] Generating benchmark candidates and predictions for {len(data.benchmark_queries):,} queries...",
        flush=True,
    )
    t0 = time.time()
    benchmark_sources = _retrieve_sources(
        data.benchmark_queries,
        data.benchmark_items,
        encoder,
        benchmark_generic,
        selected_tower,
        tower_v1_benchmark,
        k,
        encoder_batch_size=encoder_batch_size,
        category_policy=str(selection.get("category_policy", "none")),
        dense_query_prefix=dense_query_prefix,
    )
    benchmark_catboost = rank_with_selector_chunked(
        cast(pd.DataFrame, pd.concat(benchmark_sources, ignore_index=True)),
        selector,
        limit=50,
        missing_rank=k + 1,
    )
    query_map = data.benchmark_queries[["internal_query_id", "query_id"]].copy()
    scored = _frame(
        benchmark_catboost.merge(query_map, on="internal_query_id", how="left")
    )
    scored["final_score"] = scored["score"]
    predictions = rank_predictions(scored, limit=50)
    prediction_path = root / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    print(
        f"  ✓ Benchmark predictions written to {prediction_path} [{time.time() - t0:.1f}s]",
        flush=True,
    )
    data_config = _data_config(config)
    benchmark_queries_path = _data_path(
        config, data_config, "benchmark_queries", ("validation_queries", "queries")
    )
    benchmark_items_path = _data_path(
        config, data_config, "benchmark_items", ("validation_items", "items")
    )
    components = {
        "dense_model": model_name,
        "dense_model_revision": model_revision,
        "dense_model_path": str(dense_config.get("model_path", model_name)),
        "bm25_index": "lexical/bm25.npz",
        "two_tower_checkpoint": str(v1_path.relative_to(root)),
        "selector_model": str(selector_path.relative_to(root)),
        "metrics": str(metrics_path.relative_to(root)),
        "predictions": str(prediction_path.relative_to(root)),
        "benchmark_queries": str(benchmark_queries_path),
        "benchmark_items": str(benchmark_items_path),
        "category_policy": str(selection.get("category_policy", "unspecified")),
        "split": "cold-item-validation",
        "seed": config.seed,
        "retrieval_device": str(dense_config.get("device", "cpu")),
    }
    manifest = SelectedManifest(
        1,
        "PASS",
        ("bm25", "dense", "two_tower"),
        k,
        "catboost",
        "recall@50",
        components,
        config.hash,
        git_info(config.root).get("commit"),
    )
    pipeline_res = FinalPipeline(root, manifest).save_manifest(components)
    total_mins = (time.time() - t_start) / 60.0
    print("\n" + "=" * 70, flush=True)
    print(
        f"✓ All pipeline stages completed successfully in {total_mins:.1f} minutes",
        flush=True,
    )
    print(f"✓ Manifest saved to {root / 'manifest.json'}", flush=True)
    print("=" * 70, flush=True)
    return pipeline_res


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
    print(f"Generating submission: {predictions} -> {output_csv}...", flush=True)
    res = write_submission(predictions, queries, items, Path(output_csv))
    print(f"✓ Submission written to {res}", flush=True)
    return res
