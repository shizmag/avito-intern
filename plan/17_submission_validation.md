# 17 — answer.csv generation and strict round-trip validator

## Role
Вы — Data Quality/Release Engineer. Создайте единственный submission exporter и validator, который машинно доказывает соответствие формату задания.

## Goal
Преобразовать ranked predictions в `answer.csv`, перечитать файл и повторно проверить IDs, row/query coverage, whitespace list, uniqueness, corpus membership, encoding and absence of index columns.

## Context
Required columns exactly `query_id,answer`; one row per benchmark query. `answer` — whitespace-separated up to 50 item IDs. IDs are case-sensitive 16-character strings and must never pass through numeric types. Empty answer is allowed by stated 0..50 contract, но должен быть reported.

## Prerequisites
- `16` inference PASS; predictions and benchmark data fingerprints match.
- Task format in `task.md` reread before implementation; resolve delimiter/quoting semantics against original task.

## Inputs
- `predictions.parquet` from stage `16`.
- Raw/canonical benchmark queries/items.
- Explicit output path, default repo-root `answer.csv` or artifact copy plus release copy.

## Tasks
1. Validate typed predictions before formatting: all/only benchmark query IDs, known items, rank uniqueness/contiguity, <=50 unique/query, 16-char string IDs/case.
2. Left join from benchmark query order so every query appears exactly once. Sort candidates rank asc/item tie-break; join with one ASCII space, no leading/trailing/multiple whitespace.
3. Write exact columns `query_id`, `answer` via `to_csv(..., index=False, encoding="utf-8")`. Preserve empty answer as empty string through explicit read policy, not NaN confusion.
4. Implement independent read-back validator from CSV bytes/file: valid UTF-8, header exactly, no BOM/index/extra columns, expected row count, query set/uniqueness/order policy, IDs string/length/case, token whitespace, count 0..50, no duplicates, known item membership.
5. Ensure CSV parser does not infer IDs numerically: read dtype string, `keep_default_na=False`; tests include leading zeros and mixed case.
6. Write machine-readable validation report with counts, min/median/p95/max candidates, empty rows, duplicate/unknown count, SHA-256 and status. Nonzero exit on any contract violation.
7. Export command must validate automatically after write and delete/mark invalid partial output on failure; atomic rename only after validation.
8. Add mutation tests for every listed violation.

## Architecture / Design Constraints
- One canonical exporter/validator used by smoke and final release.
- Membership checked against exact benchmark item snapshot fingerprint.
- No score columns or dataframe index in CSV.
- No manual spreadsheet/open-save.
- Validator checks semantic parse and raw formatting where relevant.

## Files to Create or Modify
- `src/avito_candidate_generation/submission.py`
- `tests/test_submission.py`
- `tests/integration/test_submission_pipeline.py`
- generated `answer.csv` (gitignored) and artifact copy/report.

## Interfaces / Data Contracts
```python
def write_submission(predictions_path: Path, queries_path: Path, items_path: Path, output: Path) -> Path: ...
def validate_submission(path: Path, queries_path: Path, items_path: Path) -> ValidationReport: ...
```
ValidationReport has every check boolean/count, input/output hashes and PASS/FAIL.

## Tests
- Valid golden CSV including leading-zero/mixed-case IDs and empty answer.
- Wrong/missing/duplicate/unknown query; wrong row count.
- Numeric-coerced/15/17-char item/query IDs; unknown item; duplicate item; >50.
- Multiple/tab/newline/leading/trailing whitespace in answer.
- Extra `Unnamed: 0`, reordered/extra/missing columns, BOM/non-UTF8.
- Round-trip exporter exactly preserves IDs and emits no index.
- Tiny benchmark predictions→CSV integration.

## Verification
```bash
uv run pytest tests/test_submission.py tests/integration/test_submission_pipeline.py -q
uv run python -m avito_candidate_generation.submission write --predictions artifacts/inference/<run_id>/predictions.parquet --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet --output answer.csv
uv run python -m avito_candidate_generation.submission validate --submission answer.csv --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet
make ci
```
Required checks: columns exact; 2452 rows; query IDs exactly once; string length 16; 0..50 items; valid single-space tokenization; unique known string item IDs length 16; UTF-8/no index; read-back PASS; file SHA recorded.

## Expected Outputs / Artifacts
Validated `answer.csv`, `artifacts/submission/<run_id>/answer.csv`, `validation_report.json`, manifest/config/hash.

## Documentation Requirements
Document exact format, dtype-safe read/write, empty-answer handling, validator command and why post-write validation is mandatory.

## Definition of Done
- Mutation/unit/integration tests and `make ci` pass.
- Final write + independent reread validator exit 0.
- Report shows every named check PASS and output hash.
- Gate 5 submission-format portion PASS.

## Do Not
- Не cast IDs to numbers.
- Не hand-edit CSV.
- Не trust in-memory dataframe without read-back.
- Не silently drop unknown/duplicate items.
- Не include score/rank/index columns.

## Handoff to Next Stage
Documentation agent получает validated immutable submission/hash/report, full run lineage, ablation/test metrics and exact reproduction commands.
