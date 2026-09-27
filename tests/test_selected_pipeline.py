from __future__ import annotations

import json

import pytest

from avito_candidate_generation.pipeline import FinalPipeline, SelectedManifest


def test_selected_manifest_is_explicit_and_roundtrips(tmp_path) -> None:
    manifest = SelectedManifest(
        schema_version=1,
        status="PASS",
        selected_retrievers=("bm25", "dense", "two_tower"),
        retrieval_k=500,
        fusion="rrf",
        validation_metric="recall@50",
        components={
            "dense_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
    )
    pipeline = FinalPipeline(tmp_path, manifest)
    path = pipeline.save_manifest(
        manifest.components, config_hash="abc", git_commit="deadbeef"
    )
    loaded = FinalPipeline.load(tmp_path)
    assert path.name == "manifest.json"
    assert loaded.manifest.selected_retrievers == manifest.selected_retrievers
    assert loaded.manifest.retrieval_k == 500
    assert json.loads(path.read_text())["validation_metric"] == "recall@50"


def test_selected_manifest_rejects_ambiguous_bundle(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "PASS"})
    )
    with pytest.raises(ValueError, match="invalid selected pipeline manifest"):
        FinalPipeline.load(tmp_path)
