# 04 — Leakage-safe cold-item splits и ground truth

## Role
Вы — ML Evaluation Engineer, отвечающий за честный offline protocol и отсутствие entity leakage.

## Goal
Построить deterministic primary train/validation/test split по `item_id`, query slices и ground-truth artifacts для full-corpus retrieval evaluation.

## Context
Random row split завышает качество из-за повторяющихся query texts, items и pairs. Benchmark mostly cold-item относительно selected train IDs. Primary protocol должен моделировать unseen positive items, сохраняя item text/features доступными в searchable corpus. Train не имеет query_id, поэтому query unit — `internal_query_id`; seen/unseen slice — отдельное свойство query text относительно fold-train.

## Prerequisites
- `01`–`03` DONE; canonical unique interactions/items/queries and text representations exist.
- Duplicate pair and item snapshot policies уже зафиксированы; не переопределять.

## Inputs
- `queries_train.parquet`, `items_train.parquet`, `interactions.parquet` + manifests.
- `[splits]` config: seed, proportions default 0.8/0.1/0.1, algorithm version, minimum slice/report thresholds.

## Tasks
1. Назначить каждый unique `item_id` ровно в один split deterministic seeded hash/algorithm. Default 80/10/10; изменение требует config+report rationale.
2. Все positive interactions item следуют item split. Проверить pair uniqueness до splitting.
3. Создать ground truth per split: one row `(split, internal_query_id, relevant_item_id)` и query-level relevance counts.
4. Search corpus для primary evaluation определить явно как весь canonical train item corpus, включая cold val/test items как доступные document features и train items как distractors. Supervised retriever не видит val/test positive pairs при training.
5. Для validation/test вычислить `seen_query_text`: exact normalized `search_query` присутствует среди query texts fold-train; отдельно сохранить seen full query tuple. Primary report минимум: `seen text + cold item`, `unseen text + cold item`.
6. Добавить slices: single/multi-positive в данном ground truth, same/cross location positive, with/without search params. Slice labels не меняют split.
7. Создать optional diagnostic interaction/group holdout только отдельным artifact/config; он не участвует в final model selection по умолчанию.
8. Создать 3 item-disjoint OOF fold assignments внутри development pool для later fusion. На fixed evaluation test ничего не fit/tune. Clarify lifecycle: tune retrievers/fusion on dev/validation; test once after architecture freeze; final benchmark refit uses all train with separate 3-fold OOF generation.
9. Сохранить split summary/fingerprints и deterministic rerun check. Stratification по category допустима только item-level и с тестом disjointness; default — deterministic item hash plus distribution diagnostics.
10. CLI:
```bash
uv run python -m avito_candidate_generation.splits build --config configs/base.toml
uv run python -m avito_candidate_generation.splits verify --config configs/base.toml
```

## Architecture / Design Constraints
- Absolute invariant: train/val/test item sets pairwise disjoint.
- Query overlap допускается и измеряется; нельзя group by query так, чтобы потерять нужный seen-query+cold-item scenario.
- Full searchable corpus, не `positive + random negatives`.
- Split assignment independent of label frequency/text content unless documented stratification selected.
- Empty/rare categories report; no moving individual interactions across split for cosmetic balance.
- Ground truth excludes no difficult query silently.

## Files to Create or Modify
- `src/avito_candidate_generation/splits.py`
- `configs/validation.toml` inheriting/copying explicit base values
- `tests/test_splits.py`
- tiny split fixtures under `tests/fixtures/`

## Interfaces / Data Contracts
Artifacts:
- `item_splits.parquet`: `item_id:string`, `split: train|validation|test`, `fold:int|null`.
- `ground_truth.parquet`: `split`, `internal_query_id:string`, `item_id:string` unique.
- `query_slices.parquet`: one row/split/query with booleans/counts.
- `split_report.json`: counts, distributions, overlaps (must be zero for items), algorithm/seed/fingerprints.

OOF fold contract: every development item assigned one held-out fold; fold-train excludes every interaction for fold-heldout items.

## Tests
- Pairwise item disjointness; all interactions assigned exactly once.
- Same seed/input gives identical assignment; changed seed changes nontrivial assignments.
- Duplicate `(query,item)` rejected before split.
- Seen/unseen text slices hand-verified; query can be seen while relevant item is cold.
- Multi-positive query ground truth retained as set.
- OOF heldout item never appears in corresponding supervised fold-train positives.
- Distribution and tiny-category edge cases; no empty required split on realistic fixture/config.

## Verification
```bash
uv run pytest tests/test_splits.py -q
uv run python -m avito_candidate_generation.splits build --config configs/validation.toml
uv run python -m avito_candidate_generation.splits verify --config configs/validation.toml
make ci
```
Required assertions:
```text
train_items ∩ validation_items = ∅
train_items ∩ test_items = ∅
validation_items ∩ test_items = ∅
union(split interactions) = canonical interactions
all OOF heldout item/fold-train intersections = ∅
```
Report exact split sizes, query counts, positives/query and category/location distributions. Rerun fingerprint must match.

## Expected Outputs / Artifacts
`artifacts/splits/<run_id>/{item_splits,ground_truth,query_slices,oof_folds}.parquet`, `split_report.json`, manifest/config.

## Documentation Requirements
Explain why random row split is invalid, why corpus still includes cold item documents, difference between item coldness and query seen status, and when fixed test is allowed to be read.

## Definition of Done
- Leakage invariants and tests pass; `make ci` pass.
- Primary and OOF split artifacts deterministic/versioned.
- Both seen-query+cold-item and unseen-query+cold-item slices have reported support.
- Gate 1 split portion PASS; no metric tuning on test.

## Do Not
- Не split rows independently.
- Не evaluate on sampled 100 negatives.
- Не remove multi-positive or hard queries to improve metric.
- Не allow val/test positive pair in supervised train through duplicates.
- Не repeatedly inspect test during model selection.

## Handoff to Next Stage
Evaluator/retrievers получают immutable split and ground-truth contracts, full corpus definition, query slices and 3-fold OOF assignment.
