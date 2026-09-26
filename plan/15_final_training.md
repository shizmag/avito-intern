# 15 — Final training and artifact bundle

## Role
Вы — ML Production/Release Engineer. Refit frozen selected architecture на всех available development positives и создайте self-contained benchmark inference bundle.

## Goal
Обучить final retrievers/fusion корректно, сохранить indexes/embeddings/models/configs/versions и доказать loadability. Никакой архитектурной ретюнинговки.

## Context
После model selection все train positives могут использоваться для final retrievers. CatBoost final fusion всё равно требует 3-fold OOF training features на all development positives; нельзя заменить их in-sample scores. Затем retrieval models refit on all positives for benchmark inference. Benchmark labels недоступны.

## Prerequisites
- `14` Gate 4 PASS; frozen `configs/selected.toml` and architecture hash exist.
- All upstream implementations/tests pass; required compute/data available.

## Inputs
- Entire canonical development train queries/items/interactions.
- Selected configs for preprocessing, category strategy, sources, negative mining, union/RRF/features/CatBoost.
- Benchmark item corpus may be used only to build inference indexes/embeddings after model training; no labels.

## Tasks
1. Validate selected config hash and upstream data/preprocessing versions; refuse manual parameter overrides except resource-only batch/device settings that cannot change semantics.
2. Rebuild selected lexical index for benchmark item corpus and selected generic dense benchmark item embeddings later/within bundle as defined; corpus-specific artifacts clearly named.
3. Train final two-tower on all development positives with selected one-iteration hard-negative pipeline. Mining uses only development queries/labels and allowed item corpus; never benchmark hidden interactions.
4. Generate final 3-fold OOF candidates/features across all development positives using same selected retrievers/policies; train final CatBoost on these OOF rows. If selected fusion is RRF, skip CatBoost explicitly N/A.
5. Refit final supervised retriever after OOF fold models; fold models are not final inference model. Record parent OOF hashes.
6. Bundle preprocessing config/version, tokenizer/base model pinned revisions, checkpoints, feature dictionary, category/fallback policy, source depths, RRF k, CatBoost, ID schemas and tie-break.
7. Run fresh-process model/index load and tiny deterministic inference probe. Compare saved/reloaded outputs and validate no missing references.
8. Record training timings, seeds, dependency lock hash, git commit, data fingerprints and resource environment. Large weights/indexes remain outside git.
9. Produce artifact inventory with SHA-256 where practical and semantic fingerprints for large arrays/indexes.
10. Do not evaluate/choose on benchmark labels/submission. Final bundle status PASS only after all selected components load.

## Architecture / Design Constraints
- Training/inference feature code shared.
- Final CatBoost training remains OOF-derived.
- Resource-only changes cannot alter retrieval K/model dtype without new selected config; if precision changes semantics, it is not resource-only.
- Corpus item embeddings computed once and cache-keyed.
- Failed/partial final runs isolated; do not overwrite prior bundles.

## Files to Create or Modify
- `src/avito_candidate_generation/pipeline.py`
- `configs/final.toml` generated as explicit copy/reference of selected semantics plus final paths/resources
- `tests/integration/test_model_bundle.py`
- minimal orchestration entrypoint; reuse modules, no duplicate scripts.

## Interfaces / Data Contracts
Bundle manifest lists component role/path/hash/schema/version and exact load order. `FinalPipeline.load(bundle_dir)` (or equivalent simple function) must construct inference without training data labels. Query/item schema versions and feature order embedded.

## Tests
- Config freeze rejects semantic override.
- Tiny selected-component final fit and bundle creation.
- Fresh process load without training labels and inference equivalence.
- Missing/corrupt component/hash mismatch fails clearly.
- CatBoost training input provenance OOF; final two-tower full-data provenance.
- No benchmark query IDs/labels in training artifacts.

## Verification
```bash
uv run pytest tests/integration/test_model_bundle.py -q
uv run python -m avito_candidate_generation.pipeline fit-final --config configs/final.toml
uv run python -m avito_candidate_generation.pipeline verify-bundle --bundle artifacts/final/<run_id>
make ci
```
Assertions: all selected components and exact hashes present; final checkpoint/index/embedding dimensions valid; OOF report all folds PASS; fresh-load probe outputs same unique ranked IDs; no benchmark labels accessed.

## Expected Outputs / Artifacts
`artifacts/final/<run_id>/` containing complete manifest/config snapshot, selected retriever models/index metadata, OOF lineage, final fusion model if selected, feature dictionary, checksums and fit report.

## Documentation Requirements
Explain final refit vs OOF roles, what uses all positives, what remains fold-trained, corpus-specific caches, load command and reproducibility limits.

## Definition of Done
- Tests and `make ci` pass.
- Full selected architecture fit completes; bundle verifier exit 0.
- Fresh process can load and score without train labels.
- All metadata/seeds/revisions/fingerprints recorded; benchmark labels never used.

## Do Not
- Не change architecture after stage `14`.
- Не train CatBoost on final in-sample retriever features.
- Не overwrite previous experiment outputs.
- Не commit large model/index/embedding files.
- Не infer quality from successful fit alone.

## Handoff to Next Stage
Benchmark inference agent получает verified immutable final bundle, complete load API and corpus-specific artifact build contract.
