from avito_candidate_generation.verify import verify_repository


def test_repository_verifier_reports_unbuilt_selected_artifact(tmp_path) -> None:
    result = verify_repository(tmp_path)
    assert result["status"] == "BLOCKED"
    assert "selected_artifact" in result["blocking"]
    assert result["messages"]["selected_artifact_status"] == "NOT_RUN"


def test_selected_artifact_status_not_run(tmp_path) -> None:
    result = verify_repository(tmp_path)
    assert result["messages"]["selected_artifact_status"] == "NOT_RUN"


def test_selected_artifact_status_blocked_when_malformed(tmp_path) -> None:
    manifest = tmp_path / "artifacts/selected/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    result = verify_repository(tmp_path)
    assert result["messages"]["selected_artifact_status"] == "BLOCKED"


def test_selected_artifact_status_implemented_for_valid_manifest(tmp_path) -> None:
    manifest = tmp_path / "artifacts/selected/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"schema_version": 1, "status": "PASS", "selected_retrievers": ["bm25"], '
        '"retrieval_k": 50, "fusion": "rrf", "validation_metric": "recall@50"}'
    )
    result = verify_repository(tmp_path)
    assert result["messages"]["selected_artifact_status"] == "IMPLEMENTED"
