from pathlib import Path

from avito_candidate_generation.artifacts import make_run_context, write_manifest
from avito_candidate_generation.config import config_hash, load_config, resolve_path


def test_config_and_manifest(tmp_path: Path) -> None:
    config = load_config("configs/base.toml")
    assert config.seed == 42
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})
    assert resolve_path(config, "data/train.parquet").name == "train.parquet"
    context = make_run_context("test", "smoke", config, tmp_path)
    manifest = write_manifest(context, config, status="PASS", command="pytest")
    assert manifest.exists()
    assert manifest.read_text().find('"schema_version": 1') >= 0
