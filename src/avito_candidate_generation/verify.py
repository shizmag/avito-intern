"""Truthful repository-level deterministic verification command."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def _check_selected_config(base: Path) -> tuple[bool, str]:
    path = base / "configs/selected.toml"
    if not path.is_file():
        return False, "configs/selected.toml missing"
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
        selection = config["selection"]
        retrievers = selection["retrievers"]
        required = {"bm25", "dense", "two_tower"}
        if not required.issubset(retrievers):
            return False, "selected retriever set is incomplete"
        if int(selection["retrieval_k"]) < 1 or selection["fusion"] not in {
            "rrf",
            "catboost",
        }:
            return False, "selected retrieval policy is invalid"
        dense_model = str(config["dense"]["model"])
        if not dense_model or "fake" in dense_model.lower():
            return False, "selected dense model is placeholder"
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        return False, f"selected config invalid: {exc}"
    return True, "selected config is explicit"


def _check_python_dependencies() -> tuple[bool, str]:
    missing: list[str] = []
    for module in ("torch", "transformers", "catboost", "safetensors"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return (
        not missing,
        "all required ML dependencies import"
        if not missing
        else f"missing dependencies: {missing}",
    )


def _selected_artifact_status(base: Path) -> tuple[str, str]:
    manifest = base / "artifacts/selected/manifest.json"
    if not manifest.exists():
        return "NOT_RUN", "selected artifact manifest not generated"
    try:
        if manifest.stat().st_size == 0:
            return "BLOCKED", "selected artifact manifest is empty"
        with manifest.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        required = {
            "schema_version",
            "status",
            "selected_retrievers",
            "retrieval_k",
            "fusion",
            "validation_metric",
        }
        if not required.issubset(payload) or payload.get("status") != "PASS":
            return "BLOCKED", "selected artifact manifest is malformed"
        if not payload["selected_retrievers"] or int(payload["retrieval_k"]) < 1:
            return "BLOCKED", "selected artifact manifest has invalid retrieval policy"
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return "BLOCKED", f"selected artifact manifest unreadable: {exc}"
    return "IMPLEMENTED", "selected artifact manifest is valid"


def verify_repository(root: str | Path = ".") -> dict[str, Any]:
    base = Path(root)
    checks: dict[str, bool] = {
        path: (base / path).exists()
        for path in (
            "src/avito_candidate_generation",
            "configs/base.toml",
            "configs/dense.toml",
            "configs/selected.toml",
            "Makefile",
            "pyproject.toml",
        )
    }
    plan_files = sorted((base / "plan").glob("*.md"))
    checks["plan_00_19"] = len(plan_files) == 20 and [
        p.name[:2] for p in plan_files
    ] == [f"{i:02d}" for i in range(20)]
    selected_ok, selected_message = _check_selected_config(base)
    checks["selected_config"] = selected_ok
    dependencies_ok, dependency_message = _check_python_dependencies()
    checks["ml_dependencies"] = dependencies_ok
    artifact_status, artifact_message = _selected_artifact_status(base)
    checks["selected_artifact"] = artifact_status == "IMPLEMENTED"
    status = "PASS" if all(checks.values()) else "BLOCKED"
    return {
        "schema_version": 2,
        "status": status,
        "checks": checks,
        "messages": {
            "selected_config": selected_message,
            "ml_dependencies": dependency_message,
            "selected_artifact": artifact_message,
            "selected_artifact_status": artifact_status,
        },
        "blocking": [name for name, passed in checks.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    report = verify_repository(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
