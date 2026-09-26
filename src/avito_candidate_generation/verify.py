"""Repository-level deterministic verification command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_repository(root: str | Path = ".") -> dict[str, object]:
    base = Path(root)
    required = [
        "src/avito_candidate_generation",
        "configs/base.toml",
        "Makefile",
        "pyproject.toml",
    ]
    checks = {path: (base / path).exists() for path in required}
    plan_files = sorted((base / "plan").glob("*.md"))
    checks["plan_00_19"] = len(plan_files) == 20 and [
        p.name[:2] for p in plan_files
    ] == [f"{i:02d}" for i in range(20)]
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    report = verify_repository(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
