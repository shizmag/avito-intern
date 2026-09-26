# 01 — Repository foundation, configs и воспроизводимость

## Role
Вы — Senior Python/ML Platform Engineer. Создайте минимальный, не переусложнённый фундамент для всех последующих ML-стадий.

## Goal
Сделать репозиторий устанавливаемым, типизированным, тестируемым и config-driven; определить единый run/artifact contract и canonical CI. Retrieval/model code на этой стадии не писать.

## Context
Сейчас есть `uv.lock`, Python 3.13, `pyproject.toml` с pandas/numpy/sklearn/pyarrow/marimo, пустой `README.md`, заглушка `main.py`, `pyrightconfig.json`; `src/`, `tests/`, configs и Makefile отсутствуют. Будущие torch/sentence-transformers/CatBoost зависимости могут иметь ограничения Python 3.13 — совместимость должна проверяться, а не предполагаться.

## Prerequisites
- Прочитаны `plan/00_overview.md`, `AGENTS.md`, `pyproject.toml`, `.python-version`, `pyrightconfig.json`, `.gitignore`.
- Working tree проверен; чужие изменения не удаляются.

## Inputs
- Текущие repository metadata и lockfile.
- Cross-stage conventions из `00_overview.md`.

## Tasks
1. Создать package `src/avito_candidate_generation/` и минимальный stdlib-CLI/config/logging/artifact-manifest skeleton. Не создавать бессмысленные base classes.
2. Использовать TOML configs и `tomllib`; создать `configs/base.toml` и `configs/smoke.toml` с `seed=42`, data paths, artifact root, logging level и resource limits. Path resolution должен быть относительно repo root/config, не текущей случайной директории.
3. Добавить immutable run identity: stage + human run name + canonical config hash. В `manifest.json` писать timestamp UTC, seed, config snapshot path/hash, input fingerprints, git commit/dirty flag, Python/platform/dependency versions, command, status.
4. Добавить deterministic seed helper для `random`, NumPy и optional torch (без обязательного импорта torch). Один seed из config.
5. Настроить structured logging без секретов/row-level data. Поля: stage, run_id, rows, duration, artifact path; JSON logging не обязателен, стабильный key=value допустим.
6. Настроить `pyproject.toml`: package/build metadata, ruff, pytest, coverage, dev dependency group (`pytest`, `pytest-cov`, `ruff`, `pyright` либо принятый локальный equivalent). Не добавлять ML-heavy deps заранее.
7. Создать `Makefile`: `format-check`, `lint`, `typecheck`, `test`, `coverage`, `ci`, `smoke`; `make ci` — canonical cheap gate. Coverage thresholds должны соответствовать AGENTS.md (overall line 85%, branch 75%), но можно узко исключить пустые entrypoints.
8. Расширить `.gitignore`: `artifacts/`, model/index/cache directories, generated reports, `answer.csv`, без игнорирования source/config/tests. Не удалять raw data.
9. Проверить текущий Python 3.13. Если обязательный будущий ML stack объективно несовместим, зафиксировать проверяемый issue в `reports/environment_compatibility.md`; не менять Python version без обоснования и lock regeneration.
10. Удалить/перенаправить `main.py` только если packaging entrypoint полностью его заменяет; не ломать `eda.py`.

## Architecture / Design Constraints
- YAGNI: dataclass/TypedDict для manifest/config, stdlib argparse/logging/TOML достаточно.
- Config values проходят validation; unknown required fields и отсутствующие paths дают понятную ошибку.
- Config snapshot копируется в run directory до heavy operation.
- Artifact write для JSON/CSV metadata — atomic temp + rename, где практически полезно.
- Никакого глобального mutable config и hidden notebook state.
- Dependency versions фиксируются через `uv.lock`; большие модели не vendor/commit.

## Files to Create or Modify
- `pyproject.toml`, `uv.lock`, `.gitignore`, при необходимости `.python-version`, `pyrightconfig.json`
- `Makefile`
- `configs/base.toml`, `configs/smoke.toml`
- `src/avito_candidate_generation/__init__.py`
- `src/avito_candidate_generation/config.py`
- `src/avito_candidate_generation/reproducibility.py`
- `src/avito_candidate_generation/artifacts.py`
- `src/avito_candidate_generation/logging.py`
- `tests/test_config.py`, `tests/test_artifacts.py`, `tests/test_reproducibility.py`
- `reports/environment_compatibility.md` только при реальном blocker/решении.

## Interfaces / Data Contracts
```python
@dataclass(frozen=True)
class RunContext:
    stage: str
    run_id: str
    seed: int
    run_dir: Path
    config_path: Path
    config_hash: str
```

Manifest version `1`; обязательные keys: `schema_version`, `stage`, `run_id`, `status`, `seed`, `command`, `config`, `inputs`, `environment`, `git`, `started_at`, `finished_at`. Fingerprints: SHA-256 для малых файлов/config; для больших parquet — path, size, mtime и Parquet metadata/content strategy, позднее уточняемая data stage.

## Tests
- TOML config loads, merges only documented base/override semantics, rejects missing/invalid seed/path.
- Same canonical config yields same hash/run identity component; key ordering does not change hash.
- Seed helper repeats NumPy/Python sequences.
- Manifest round-trip and required fields; failed run records `FAIL` and diagnostic.
- Artifact root cannot accidentally resolve to raw data path.

## Verification
```bash
uv sync --all-groups
uv run pytest tests/test_config.py tests/test_artifacts.py tests/test_reproducibility.py -q
uv run ruff check .
uv run pyright
make ci
uv run python -c "import avito_candidate_generation"
```
Expected: all exit 0; two runs with same config have identical config hash; no generated artifact tracked by git.

## Expected Outputs / Artifacts
- Installable package and lockfile.
- Stable config/run manifest helpers.
- Canonical CI command.
- No trained model/index/candidate artifact yet.

## Documentation Requirements
Document config precedence, artifact directory convention, determinism limits (especially future GPU), and why versions/fingerprints are recorded. Comments explain non-obvious reproducibility choices only.

## Definition of Done
- Targeted tests, typecheck, lint, coverage and `make ci` pass.
- `uv sync --all-groups` works from clean environment.
- Config and manifest public types are typed and tested.
- Existing EDA remains runnable/import-safe.
- No retrieval or benchmark-specific logic introduced.

## Do Not
- Не добавлять Hydra/MLflow/DI framework ради инфраструктуры.
- Не добавлять torch/CatBoost/sentence-transformers до стадии, где они нужны.
- Не хардкодить absolute local paths, benchmark IDs или machine-specific devices.
- Не заявлять GPU determinism без теста/оговорки.

## Handoff to Next Stage
Следующий агент получает работающие `uv`, package, config loader, run contexts, artifact manifests, logging и `make ci`. Он может сосредоточиться на данных и executable EDA.
