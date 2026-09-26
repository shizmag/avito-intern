"""Minimal immutable final bundle loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FinalPipeline:
    def __init__(self, bundle_dir: Path, manifest: dict[str, Any]) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = manifest

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "FinalPipeline":
        path = Path(bundle_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
            raise ValueError("invalid final bundle manifest")
        return cls(path, manifest)

    def save_manifest(self, components: list[dict[str, Any]]) -> Path:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        target = self.bundle_dir / "manifest.json"
        payload = {"schema_version": 1, "status": "PASS", "components": components}
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target
