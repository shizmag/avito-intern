"""Run identity and manifest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .config import Config


@dataclass(frozen=True)
class RunContext:
    stage: str
    run_id: str
    seed: int
    run_dir: Path
    config_path: Path
    config_hash: str


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    result: dict[str, Any] = {
        "path": str(p),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 10_000_000:
        result["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def git_info(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty}


def make_run_context(
    stage: str, run_name: str, config: Config, artifact_root: str | Path | None = None
) -> RunContext:
    if not stage or not run_name:
        raise ValueError("stage and run_name are required")
    root = Path(artifact_root) if artifact_root else config.root / "artifacts"
    run_id = f"{stage}-{run_name}-{config.hash[:12]}"
    run_dir = (root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(stage, run_id, config.seed, run_dir, config.path, config.hash)


def write_manifest(
    context: RunContext,
    config: Config,
    *,
    status: str,
    command: str,
    inputs: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "stage": context.stage,
        "run_id": context.run_id,
        "status": status,
        "seed": context.seed,
        "command": command,
        "config": {
            "path": str(config.path),
            "hash": config.hash,
            "snapshot": config.values,
        },
        "inputs": inputs or {},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "git": git_info(config.root),
        "started_at": started_at or now,
        "finished_at": finished_at or now,
    }
    target = context.run_dir / "manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return target
