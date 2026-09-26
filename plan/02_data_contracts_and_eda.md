# 02 — Data contracts, canonical tables и воспроизводимый EDA

## Role
Вы — Senior Data/ML Engineer, отвечающий за schema contracts, ID integrity и автоматическую проверку наблюдений EDA.

## Goal
Создать validated canonical data layer и machine-readable EDA report. Превратить предположения о данных в проверки, не обучая модели.

## Context
Raw files: `train.parquet` (~497673 rows), `benchmark_queries.parquet` (2452), `benchmark_items.parquet` (189212). Train содержит query+item positive interactions; benchmark labels скрыты. Existing `eda.py` показывает ~74529 unique query texts, ~344825 train items, 907 benchmark query texts seen in train, 18142 benchmark items selected in train, category match ~0.9999+, location match ~0.84 и сильное lexical coverage. Эти числа нужно пересчитать executable command. Raw train может содержать duplicate rows/pairs и несколько snapshots одного item.

## Prerequisites
- `01_repository_foundation.md` DONE; `make ci` pass.
- Прочитаны `task.md`, весь `eda.py`, actual Arrow schemas и config conventions.

## Inputs
- `data/train.parquet`
- `data/benchmark_queries.parquet`
- `data/benchmark_items.parquet`
- `configs/base.toml`, `configs/smoke.toml`

## Tasks
1. Описать schema contracts для трех raw datasets: required columns, Arrow/pandas-compatible types, nullability policy, finite/range constraints и primary IDs.
2. Загружать только требуемые columns где возможно. На boundary принудительно проверять, что `item_id`/`query_id` являются string и имеют length 16; никогда не исправлять numeric coercion постфактум.
3. Проверить uniqueness benchmark `query_id`, benchmark `item_id`, nonempty datasets, no missing IDs, boolean-like flags, coordinate/rating/price validity. Invalid domain values: fail для IDs/schema; report+configured policy для nullable numerical fields.
4. Deduplicate exact duplicate interactions и duplicate `(internal_query_id,item_id)` deterministically. Сохранить raw count, dropped count. Не считать дубликаты дополнительным relevance weight по умолчанию.
5. Создать `internal_query_id` как stable, versioned SHA-256-based identifier полного canonical query tuple: raw `search_query`, normalized scalar representations of location, delivery flag, params, category. Не использовать Python `hash()` и не подменять benchmark `query_id`.
6. Построить canonical train query table, positive interaction table и train item corpus. Для нескольких snapshots одного `item_id` без timestamp выбрать детерминированно modal exact item-feature row; tie-break — canonical serialized row lexicographically. Report conflict rates per field. Не использовать label frequency как item feature.
7. Benchmark tables оставить отдельными; не merge train item history в benchmark item features по умолчанию.
8. Перенести вычисления ключевого EDA из marimo в CLI/report function: counts, nulls, duplicates, ID overlaps, exact query-text overlap, category/location match on positives, phrase/token field coverage, price/rating/geo diagnostics. `eda.py` может вызывать reusable functions, но не быть единственной реализацией.
9. Сохранить canonical Parquet с explicit schemas и deterministic sort; создать data fingerprint из schema, row count и stable content hash strategy. Для полного hash разрешён streaming Arrow batches.
10. Добавить CLI:
```bash
uv run python -m avito_candidate_generation.data validate --config configs/base.toml
uv run python -m avito_candidate_generation.data build-canonical --config configs/base.toml
uv run python -m avito_candidate_generation.eda report --config configs/base.toml
```

## Architecture / Design Constraints
- Raw files read-only; canonical outputs under `artifacts/data/<run_id>/`.
- Query/item tables разделены от interactions.
- Decimal price не переводить неявно в lossy float при ingestion; cleaning/transform — feature stage.
- Category equality — empirical statistic, не schema invariant.
- Location equality — statistic, не hard filter.
- Historical popularity/CTR не создавать в core data layer.
- Fail loudly с counts/examples redacted to bounded rows; не логировать весь private text.
- Full mode reproducible; smoke mode uses deterministic hash/sample, не `head()` только из одного category.

## Files to Create or Modify
- `src/avito_candidate_generation/schemas.py`
- `src/avito_candidate_generation/data.py`
- `src/avito_candidate_generation/eda.py`
- при необходимости обновить root `eda.py` для reuse
- `configs/base.toml`, `configs/smoke.toml` data sections
- `tests/test_schemas.py`, `tests/test_data.py`, `tests/test_eda.py`
- tiny synthetic parquet fixtures under `tests/fixtures/`

## Interfaces / Data Contracts
Canonical outputs:
- `queries_train.parquet`: one row/internal_query_id, all raw query fields.
- `items_train.parquet`: one row/string item_id, canonical raw item fields.
- `interactions.parquet`: unique `(internal_query_id,item_id)`, label implicit positive=1.
- benchmark tables preserve exact `query_id` and `item_id` case.
- `data_report.json`: schema version, counts, duplicates, overlap, match/coverage rates, null/domain diagnostics, fingerprints.

`internal_query_id` algorithm/version must be public metadata. If same query tuple appears twice, ID identical; any field change changes ID with overwhelming probability.

## Tests
- Missing/wrong column/type, null ID, 15/17-char ID, duplicate benchmark ID fail.
- Leading zeros and case in 16-char IDs survive parquet→canonical→parquet.
- Exact pair dedup and stable item snapshot tie-break.
- Internal query ID deterministic and sensitive to each tuple field.
- Null text allowed per contract; broken numeric values reported/handled by configured policy.
- EDA statistics on hand-calculated fixture.
- Full raw validation test may be marked data integration, not required when raw files absent in CI.

## Verification
```bash
uv run pytest tests/test_schemas.py tests/test_data.py tests/test_eda.py -q
uv run python -m avito_candidate_generation.data validate --config configs/base.toml
uv run python -m avito_candidate_generation.data build-canonical --config configs/base.toml
uv run python -m avito_candidate_generation.eda report --config configs/base.toml
make ci
```
Assertions:
- raw row counts equal 497673 / 2452 / 189212 for provided snapshot;
- benchmark IDs unique and length 16;
- canonical interaction pairs unique;
- canonical item IDs unique;
- report reproduces approximate known EDA and emits exact measured values, not hardcoded expected metrics;
- all output manifests reference input fingerprints.

## Expected Outputs / Artifacts
`artifacts/data/<run_id>/{queries_train,items_train,interactions,benchmark_queries,benchmark_items}.parquet`, `data_report.json`, `manifest.json`, config snapshot.

## Documentation Requirements
Explain item snapshot policy, interaction dedup semantics, query identity vs query-text seen slice, and why category/location observations are not guaranteed invariants. Keep existing EDA useful by pointing it to canonical computations.

## Definition of Done
- Schema and ID tests pass; `make ci` pass.
- Canonical artifacts build deterministically twice with same fingerprints.
- EDA report contains all known observations plus denominators.
- No raw file modified; no labels/features leaked into benchmark tables.
- Gate 1 data-contract portion PASS in manifest.

## Do Not
- Не cast IDs through int/float.
- Не silently fill missing text/numerics in raw canonical tables.
- Не select item snapshot by future/unknown timestamp or positive frequency.
- Не смешивать train selected-item popularity с benchmark features.
- Не оставлять критичные checks только в notebook/marimo UI.

## Handoff to Next Stage
Агенты preprocessing и splits получают versioned canonical query/item/interaction tables, stable IDs, exact schemas, data fingerprints и reproducible EDA report.
