"""Explicit offline model provisioning and data preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
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


def prepare(
    config_path: str | Path = "configs/selected.toml", *, allow_network: bool = False
) -> dict[str, Any]:
    """Validate data and provision configured transformer model explicitly."""
    config = load_config(config_path)
    data_report = validate_from_config(config.path)
    dense = config.values.get("dense", {})
    if not isinstance(dense, dict):
        raise ValueError("selected config requires [dense]")
    model_id = str(dense.get("model", ""))
    revision = str(dense.get("revision", ""))
    if not model_id or not revision or revision == "local-cache-required":
        raise ValueError("dense model requires model ID and immutable revision")
    configured = dense.get("model_path")
    if not isinstance(configured, str) or not configured:
        raise ValueError("dense.model_path is required; see README")
    model_path = Path(configured).expanduser()
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
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "model": model_id,
        "revision": revision,
        "path": str(model_path),
        "files": files,
        "data": data_report,
        "config_hash": config.hash,
    }
    _write_json(manifest_path, payload)
    return {
        "status": "PASS",
        "model": model_id,
        "revision": revision,
        "model_path": str(model_path),
        "model_manifest": str(manifest_path),
        "data": data_report,
    }
