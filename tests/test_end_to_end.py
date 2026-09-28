from avito_candidate_generation.end_to_end import run_end_to_end


def test_end_to_end_manifest_reports_blocked_stages(tmp_path) -> None:
    result = run_end_to_end(tmp_path)
    assert result["status"] == "BLOCKED"
    stages = result["stages"]
    assert isinstance(stages, dict)
    assert stages["selected_pipeline"] == "NOT_RUN"


def test_end_to_end_reports_blocked_malformed_artifact(tmp_path) -> None:
    manifest = tmp_path / "artifacts/selected/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    result = run_end_to_end(tmp_path)
    assert result["stages"]["selected_pipeline"] == "BLOCKED"
