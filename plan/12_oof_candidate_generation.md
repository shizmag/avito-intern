# 12 — Leakage-safe OOF candidate and feature generation

## Role
Вы — ML Pipeline Engineer specializing in stacking/OOF correctness. Постройте training table для fusion без retriever leakage.

## Goal
Для каждого из 3 OOF folds обучить supervised retrievers только на other folds, retrieve held-out queries against realistic corpus, build union features, then label only naturally retrieved pairs. Получить audited OOF CatBoost dataset.

## Context
CatBoost не может учиться на scores retriever, который видел те же held-out positive pairs. Lexical/generic dense are unsupervised and may index item documents, но supervised two-tower per fold должен исключать heldout positives/items from training interactions. Positive injection в candidate pool искусственно завышает selector training и запрещена.

## Prerequisites
- `04` OOF fold contract; `05` evaluator; retriever implementations/configs `06`–`10`; union/features `11` DONE.
- Selected validation architecture components known, но CatBoost ещё не обучен.

## Inputs
- `oof_folds.parquet`, canonical interactions/query/item corpus.
- Frozen retriever and feature configs.
- Per-fold resource config and manifests.

## Tasks
1. Orchestrate exactly 3 folds by default. For fold f: supervised retriever train interactions exclude all heldout fold positives/items per split contract; fit trainable preprocessing only on fold-train; fit/train two-tower and mine fold-safe negatives if selected.
2. Unsupervised BM25/generic pretrained dense may index complete searchable item text corpus because item documents are available at retrieval time; they must not consume heldout labels. Reuse immutable embeddings/index only if manifests prove this.
3. Generate source top-K for heldout query population against same realistic full corpus definition. Query may have multiple heldout positives.
4. Build union/pair features with stage `11`. Join labels: 1 iff `(internal_query_id,item_id)` in heldout ground truth, otherwise sampled/retrieved weak negative 0.
5. Never append a missing positive to candidates. Report candidate coverage: positive pairs retrieved/total and query Recall@pool; this is upper bound for fusion.
6. Ensure each OOF candidate row/query belongs to exactly one heldout fold; no duplicate pair across folds unless query appears in multiple folds due different cold items — then include explicit `fold` and define row/group semantics. Prefer item-fold ground truth and stable pair uniqueness `(fold,query,item)`.
7. Persist fold models/candidates only as needed; cache by fold config/input hash and support resume after complete manifest. Partial fold never marked PASS.
8. Validate label distribution, positives/query, negatives/positive, source coverage and no fold-train positive collision. Do not balance by dropping difficult/zero-positive-retrieved queries from coverage report; CatBoost table naturally has no positive for missed queries and report makes this explicit.
9. Optionally downsample easy retrieved negatives only after feature construction using deterministic source/rank-stratified policy; keep hard top candidates and record sampling probability. Default bounded union may be manageable; measure first.
10. Add orchestrator CLI and completion manifest requiring all folds PASS.

## Architecture / Design Constraints
- OOF correctness more important than compute reuse.
- Models trained for fold f cannot see heldout positive pairs; indexing heldout item text is allowed and required to retrieve cold items.
- Feature functions identical to inference.
- Group column `internal_query_id`/fold retained for ranker option and group-safe diagnostics.
- Fixed test excluded entirely.
- Resource-aware: folds sequential by default, explicit parallel option only with independent devices/RAM.

## Files to Create or Modify
- `src/avito_candidate_generation/oof.py`
- `configs/oof.toml`
- `tests/test_oof.py`
- `tests/integration/test_oof_pipeline.py`

## Interfaces / Data Contracts
OOF table: stage `11` pair features + `fold:int8`, `label:int8`, and optional `sampling_weight`; unique `(fold,internal_query_id,item_id)`. Adjacent `oof_report.json` has per-fold train/heldout item/pair hashes, retriever model IDs, pool recall and class counts.

## Tests
- Fake trainable retriever records IDs and proves heldout positives absent from fit input.
- Heldout item documents remain searchable.
- No positive injection: intentionally missed positive stays absent and lowers pool recall.
- Label join handles multi-positive query and duplicate source candidates.
- Every fold heldout assignment covered exactly once; fixed test absent.
- Resume rejects partial/stale fold.
- Tiny 3-fold end-to-end generates finite features and valid labels.

## Verification
```bash
uv run pytest tests/test_oof.py tests/integration/test_oof_pipeline.py -q
uv run python -m avito_candidate_generation.oof build --config configs/oof.toml
uv run python -m avito_candidate_generation.oof verify --config configs/oof.toml
make ci
```
Per-fold assertions: heldout positive pairs/items absent from supervised fit; candidate schema valid; labels agree exactly with ground truth intersection; no injected rows; pool Recall@K reported; all three fold manifests PASS. Global OOF artifact class counts and fingerprints stable on rerun.

## Expected Outputs / Artifacts
`artifacts/oof/<run_id>/fold_<n>/...`, `oof_candidates_features.parquet`, `oof_report.json`, per-fold metrics/manifests and global manifest/config.

## Documentation Requirements
Explain why OOF is mandatory, distinction between item-document indexing and label training leakage, no-positive-injection policy, fold/query duplication semantics and compute/cache trade-offs.

## Definition of Done
- Unit/integration tests, OOF verifier and `make ci` pass.
- Three complete fold-safe datasets generated or explicit resource blocker reported; incomplete means stage not DONE.
- OOF pool recall and class balance known.
- Fixed test untouched; all feature schemas match inference.

## Do Not
- Не train one two-tower on all positives then call its training candidates OOF.
- Не append positives missing from retrieval.
- Не evaluate fusion potential on sampled random corpus.
- Не mix fixed test rows.
- Не hide folds/queries with zero retrieved positives.

## Handoff to Next Stage
CatBoost agent получает leakage-audited OOF pair table, fixed feature dictionary, fold/group metadata, pool recall upper bound and RRF baseline.
