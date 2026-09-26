"""TOML configuration loading and canonical hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """Raised for invalid pipeline configuration."""


@dataclass(frozen=True)
class Config:
    values: dict[str, Any]
    path: Path
    root: Path
    hash: str

    @property
    def seed(self) -> int:
        value = self.values.get("seed")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError("seed must be non-negative integer")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(values: dict[str, Any]) -> str:
    payload = json.dumps(
        _canonical(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open("rb") as fh:
        values = tomllib.load(fh)
    seed = values.get("seed")
    project = values.get("project")
    if seed is None and isinstance(project, dict):
        seed = project.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            values["seed"] = seed
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("config requires integer seed")
    root_value = values.get("repo_root")
    root = (
        (config_path.parent / root_value).resolve()
        if isinstance(root_value, str)
        else config_path.parent.parent
    )
    return Config(values=values, path=config_path, root=root, hash=config_hash(values))


def resolve_path(config: Config, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config.root / path).resolve()
