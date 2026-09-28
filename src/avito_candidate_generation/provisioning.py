"""Explicit offline model provisioning and data preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import Config, load_config, resolve_path
from .data import validate_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "model_manifest.json"
    ]


def _has_required_model_files(path: Path) -> bool:
    names = {file.name for file in path.rglob("*") if file.is_file()}
    has_tokenizer = bool(names & {"tokenizer.json", "tokenizer.model", "spiece.model"})
    has_weights = bool(names & {"model.safetensors", "pytorch_model.bin"})
    return "config.json" in names and has_tokenizer and has_weights


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def provision_model_artifact(
    model_id: str,
    revision: str,
    configured_path: str,
    config: Config,
    *,
    allow_network: bool = False,
    trust_remote_code: bool = False,
    remote_code_reason: str = "",
) -> dict[str, Any]:
    """Ensure a model directory exists, contains required files, and has a valid manifest."""
    if not model_id or not revision or revision == "local-cache-required":
        raise ValueError("model requires model ID and immutable revision")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("model path is required")

    model_path = Path(configured_path).expanduser()
    if not model_path.is_absolute():
        model_path = resolve_path(config, model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = model_path / "model_manifest.json"

    if not _has_required_model_files(model_path):
        try:
            from huggingface_hub import (  # type: ignore[import-not-found]
                snapshot_download,
            )
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for model provisioning; "
                "provide a complete local model directory instead"
            ) from exc
        if not allow_network:
            raise RuntimeError(
                "model is not provisioned locally; rerun with --allow-network or "
                f"place a complete offline model at {model_path}"
            )
        try:
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(model_path),
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "unable to provision model; check network access or place a "
                f"complete offline model at {model_path}"
            ) from exc
    if not _has_required_model_files(model_path):
        raise RuntimeError(f"incomplete model artifact: {model_path}")

    files = _model_files(model_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "model": model_id,
        "revision": revision,
        "local_path": str(model_path),
        "files": files,
        "n_files": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    if trust_remote_code:
        payload["trust_remote_code"] = True
        payload["reason_for_remote_code"] = remote_code_reason
    _write_json(manifest_path, payload)
    return payload


def prepare(
    config_path: str | Path = "configs/selected.toml", *, allow_network: bool = False
) -> dict[str, Any]:
    """Validate data and provision configured transformer models explicitly."""
    config = load_config(config_path)
    data_report = validate_from_config(config.path)
    dense = config.values.get("dense", {})
    if not isinstance(dense, dict):
        raise ValueError("selected config requires [dense]")
    dense_id = str(dense.get("model", ""))
    dense_revision = str(dense.get("revision", ""))
    dense_path = str(dense.get("model_path", ""))

    dense_manifest = provision_model_artifact(
        dense_id,
        dense_revision,
        dense_path,
        config,
        allow_network=allow_network,
    )

    reranker = config.values.get("reranker")
    reranker_manifest = None
    if isinstance(reranker, dict):
        r_id = str(reranker.get("model_id", reranker.get("model", "")))
        r_rev = str(reranker.get("revision", ""))
        r_path = str(reranker.get("local_path", reranker.get("model_path", "")))
        if r_id and r_rev and r_path:
            reranker_manifest = provision_model_artifact(
                r_id,
                r_rev,
                r_path,
                config,
                allow_network=allow_network,
                trust_remote_code=bool(reranker.get("trust_remote_code", True)),
                remote_code_reason=str(
                    reranker.get(
                        "remote_code_reason",
                        "Custom XLM-RoBERTa architecture with Flash Attention",
                    )
                ),
            )

    return {
        "schema_version": 1,
        "status": "PASS",
        "data_report": data_report,
        "model_manifest": dense_manifest,
        "reranker_manifest": reranker_manifest,
    }
