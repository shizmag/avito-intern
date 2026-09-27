"""Selected pipeline manifest and deterministic retrieval orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import validate_candidates
from .fusion.rrf import reciprocal_rank_fusion
from .retrievers.bm25 import retrieve_bm25
from .retrievers.dense import EmbeddingArtifact, TextEncoder, encode_texts
from .retrievers.exact_search import exact_top_k
from .retrievers.two_tower import TwoTowerModel

_REQUIRED_MANIFEST = {
    "schema_version",
    "status",
    "selected_retrievers",
    "retrieval_k",
    "fusion",
    "validation_metric",
}


@dataclass(frozen=True)
class SelectedManifest:
    schema_version: int
    status: str
    selected_retrievers: tuple[str, ...]
    retrieval_k: int
    fusion: str
    validation_metric: str
    components: dict[str, Any]
    config_hash: str | None = None
    git_commit: str | None = None

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "selected_retrievers": list(self.selected_retrievers),
            "retrieval_k": self.retrieval_k,
            "fusion": self.fusion,
            "validation_metric": self.validation_metric,
            "components": self.components,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
        }


class FinalPipeline:
    def __init__(
        self, bundle_dir: Path, manifest: SelectedManifest | dict[str, Any]
    ) -> None:
        self.bundle_dir = bundle_dir
        if isinstance(manifest, SelectedManifest):
            self.manifest = manifest
        else:
            try:
                self.manifest = SelectedManifest(
                    schema_version=int(manifest.get("schema_version", 1)),
                    status=str(manifest.get("status", "PASS")),
                    selected_retrievers=("legacy",),
                    retrieval_k=1,
                    fusion="legacy",
                    validation_metric="legacy",
                    components=dict(manifest.get("components", {})),
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("invalid pipeline manifest") from exc

    @classmethod
    def load(cls, bundle_dir: str | Path) -> FinalPipeline:
        path = Path(bundle_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _REQUIRED_MANIFEST.issubset(payload):
                selected = tuple(str(x) for x in payload["selected_retrievers"])
                retrieval_k = int(payload["retrieval_k"])
                if retrieval_k < 1 or not selected:
                    raise ValueError("selected manifest has invalid retrieval policy")
                manifest: SelectedManifest | dict[str, Any] = SelectedManifest(
                    schema_version=int(payload["schema_version"]),
                    status=str(payload["status"]),
                    selected_retrievers=selected,
                    retrieval_k=retrieval_k,
                    fusion=str(payload["fusion"]),
                    validation_metric=str(payload["validation_metric"]),
                    components=dict(payload.get("components", {})),
                    config_hash=payload.get("config_hash"),
                    git_commit=payload.get("git_commit"),
                )
            elif (
                payload.get("schema_version") == 1
                and payload.get("status") == "PASS"
                and "components" in payload
            ):
                manifest = SelectedManifest(
                    schema_version=1,
                    status="PASS",
                    selected_retrievers=("legacy",),
                    retrieval_k=1,
                    fusion="legacy",
                    validation_metric="legacy",
                    components=dict(payload["components"]),
                )
            else:
                raise ValueError("selected manifest is incomplete or not PASS")
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid selected pipeline manifest: {manifest_path}"
            ) from exc
        return cls(path, manifest)

    def save_manifest(
        self, components: list[dict[str, Any]] | dict[str, Any], **metadata: Any
    ) -> Path:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        target = self.bundle_dir / "manifest.json"
        payload = (
            self.manifest.to_dict()
            if isinstance(self.manifest, SelectedManifest)
            else dict(self.manifest)
        )
        payload.setdefault("schema_version", 1)
        payload.setdefault("status", "PASS")
        payload["components"] = components
        payload.update(metadata)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return target


def build_selected_candidates(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    dense_encoder: TextEncoder | None = None,
    item_embeddings: EmbeddingArtifact | None = None,
    two_tower: TwoTowerModel | None = None,
    retrieval_k: int = 500,
    rrf_k: int = 60,
    category_policy: str = "none",
) -> pd.DataFrame:
    """Build selected BM25+dense union and RRF scores; no fallback algorithms."""
    required_query = {"internal_query_id", "text"}
    required_item = {"item_id", "text"}
    if not required_query.issubset(queries.columns) or not required_item.issubset(
        items.columns
    ):
        raise ValueError(
            "selected retrieval requires internal_query_id/text and item_id/text"
        )
    bm25 = retrieve_bm25(queries, items, k=retrieval_k, category_policy=category_policy)
    sources = [bm25]
    if dense_encoder is not None:
        if item_embeddings is None:
            raise ValueError(
                "dense item embeddings are required when dense encoder is selected"
            )
        query_embeddings = encode_texts(
            dense_encoder, queries["text"].astype(str).tolist(), batch_size=32
        )
        dense = exact_top_k(
            query_embeddings,
            item_embeddings.embeddings,
            list(item_embeddings.item_ids),
            k=retrieval_k,
            query_ids=queries["internal_query_id"].astype(str).tolist(),
        )
        sources.append(dense)
    if two_tower is not None:
        query_embeddings = two_tower.encode_queries(
            queries["text"].astype(str).tolist()
        )
        item_embeddings_tower = two_tower.encode_items(
            items["text"].astype(str).tolist()
        )
        two_tower_candidates = exact_top_k(
            query_embeddings,
            item_embeddings_tower,
            items["item_id"].astype(str).tolist(),
            k=retrieval_k,
            query_ids=queries["internal_query_id"].astype(str).tolist(),
            source="two_tower",
        )
        sources.append(two_tower_candidates)
    candidates = pd.concat(sources, ignore_index=True)
    validate_candidates(candidates)
    return reciprocal_rank_fusion(candidates, rrf_k=rrf_k, limit=retrieval_k)


def compose_retrieval_tables(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    *,
    query_columns: tuple[str, ...] = (
        "search_query",
        "search_infm_params_text",
        "search_category",
    ),
    item_columns: tuple[str, ...] = (
        "item_title_raw",
        "item_infm_params_text",
        "item_description_raw",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic query/item text tables from canonical Avito fields."""
    if "internal_query_id" not in queries or "item_id" not in items:
        raise ValueError("canonical tables require internal_query_id and item_id")
    missing_query = sorted(set(query_columns).difference(queries.columns))
    missing_item = sorted(set(item_columns).difference(items.columns))
    if missing_query or missing_item:
        raise ValueError(
            f"missing retrieval fields: queries={missing_query}, items={missing_item}"
        )
    query_text = (
        queries[list(query_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    item_text = (
        items[list(item_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return (
        pd.DataFrame(
            {
                "internal_query_id": queries["internal_query_id"].astype(str),
                "text": query_text,
            }
        ),
        pd.DataFrame({"item_id": items["item_id"].astype(str), "text": item_text}),
    )


def write_selected_candidates(
    candidates: pd.DataFrame,
    output: str | Path,
    *,
    manifest: FinalPipeline,
) -> Path:
    required = {
        "internal_query_id",
        "item_id",
        "source",
        "score",
        "rank",
        "rrf_score",
        "retriever_count",
    }
    if not required.issubset(candidates.columns):
        raise ValueError(
            f"selected candidates missing columns: {sorted(required.difference(candidates.columns))}"
        )
    validate_candidates(candidates)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    candidates.sort_values(
        ["internal_query_id", "rank", "item_id"], kind="mergesort"
    ).to_parquet(target, index=False)
    return target
