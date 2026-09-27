"""End-to-end repository contract verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .verify import verify_repository


def run_end_to_end(root: str | Path = ".") -> dict[str, Any]:
    result = verify_repository(root)
    checks = result["checks"]
    artifact_status = str(result["messages"].get("selected_artifact_status", "NOT_RUN"))
    selected_ready = artifact_status == "IMPLEMENTED"
    result["stages"] = {
        "data_contract": "PASS"
        if checks.get("src/avito_candidate_generation")
        else "BLOCKED",
        "selected_config": "PASS" if checks.get("selected_config") else "BLOCKED",
        "selected_pipeline": artifact_status,
        "submission": "PASS" if selected_ready else artifact_status,
    }
    result["status"] = "PASS" if result.get("status") == "PASS" else "BLOCKED"
    return result


def main() -> None:
    print(json.dumps(run_end_to_end(), indent=2))


if __name__ == "__main__":
    main()
