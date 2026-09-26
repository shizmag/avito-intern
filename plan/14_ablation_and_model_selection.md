# 14 — Controlled ablations и architecture selection

## Role
Вы — Lead ML Scientist/Reviewer. Проведите ограниченный набор сопоставимых ablations, зафиксируйте final architecture до test evaluation и запретите post-hoc drift.

## Goal
Ответить, какие компоненты реально дают Recall@50/coverage; выбрать simplest winning config и один раз подтвердить его на cold-item test.

## Context
Нельзя включать компоненты «потому что принято». Обязательная лестница: simple baseline при смысле, BM25 raw, +lemma, generic dense, in-batch two-tower, hard-negative two-tower, lexical+dense RRF, all-source RRF, CatBoost, optional source only when oracle headroom. Validation — tuning; fixed test — final unbiased check после freeze.

## Prerequisites
- `01`–`13` DONE; Gate 1–3 PASS and RRF/CatBoost validation results ready.
- All experiment metrics share compatible data/split/corpus fingerprints.

## Inputs
- Machine-readable metrics/candidates/manifests from stages 06–13.
- Resource/timing stats and slice reports.
- Acceptance config with minimum meaningful improvement and maximum slice regression defined before looking at test.

## Tasks
1. Build experiment registry/table, reject incomparable runs (different ground truth/corpus/query population) or label them non-comparable.
2. Ensure following rows or explicit N/A rationale: popularity/simple safe baseline; BM25 raw; BM25 lemma; generic dense; two-tower in-batch; hard-negative two-tower; BM25+dense RRF; all-selected retriever RRF; CatBoost; optional char n-gram TF-IDF only if lexical error analysis/oracle gap warrants it.
3. For each report R@10/20/50/100/200/500 where ranking depth supports it, pool/oracle recall, slices, latency/build time/memory and delta vs preceding/strong baseline.
4. Run bounded parameter studies only: source top-K/candidate pool size, field weights, RRF k, category global/hard/fallback, lemma, model choice already bounded, two-tower hard negatives, CatBoost depth/iterations. No broad grid; log all tried configs.
5. Diagnose bottleneck: retrieval gap (`pool R@500`) vs selection gap (`pool coverage - final R@50`). Decide whether extra source or fusion work justified.
6. Check unexpected gains for leakage/population/duplicate changes before accepting.
7. Select final validation config by R@50, with predefined slice/resource tie-break. Prefer simpler RRF if CatBoost gain below threshold; drop sources that add no union relevant hits or hurt selection/cost.
8. Freeze config and hashes read-only/conventionally immutable. Then run exactly one canonical fixed cold-item test evaluation. Do not retune after seeing test; discrepancies go to limitations.
9. Add uncertainty where feasible: paired query bootstrap CI for validation difference with fixed seed. This is optional for model fit but required before calling tiny delta meaningful.
10. Generate decision report and machine-readable selected architecture manifest; mark Gate 4 PASS/FAIL.

## Architecture / Design Constraints
- Primary metric validation Recall@50; test not tuner.
- Submission attempts not experiment rows for local tuning.
- Same query-level bootstrap pairs methods on identical population.
- Optional char TF-IDF is separate experiment and only enters final if incremental oracle/final evidence.
- Production architecture discussion does not alter benchmark exact-search choice.

## Files to Create or Modify
- `src/avito_candidate_generation/experiments.py`
- `configs/ablation.toml`
- generated `configs/selected.toml` (complete frozen config, not opaque inheritance)
- `tests/test_experiments.py`
- report templates/utilities as needed.

## Interfaces / Data Contracts
Experiment registry row: run ID, config/data/split/corpus hashes, enabled components, metrics, slice metrics, resources, status, comparable group. `selected_architecture.json`: chosen sources/policies/models/features/fusion, exact artifact/config hashes, validation result, acceptance rationale, test result added only after freeze.

## Tests
- Registry rejects metric with incompatible fingerprints.
- Delta and paired bootstrap deterministic/hand-checkable.
- Selection tie-break follows configured policy.
- Frozen selected config complete and parseable; referenced artifacts exist.
- Test evaluation guard requires frozen config and records exactly one canonical run.
- Optional experiment N/A rationale represented, not silently missing.

## Verification
```bash
uv run pytest tests/test_experiments.py -q
uv run python -m avito_candidate_generation.experiments summarize --config configs/ablation.toml
uv run python -m avito_candidate_generation.experiments freeze --config configs/ablation.toml --output configs/selected.toml
uv run python -m avito_candidate_generation.pipeline evaluate --config configs/selected.toml --split test --allow-test-evaluation
make ci
```
Expected table has required rows/NAs, deltas, slices, oracle/selection gaps and resources. Frozen hash recorded before test. Any second test attempt without explicit audited override fails.

## Expected Outputs / Artifacts
`artifacts/experiments/<run_id>/ablation.csv`, `ablation.md`, `decision_report.md`, `selected_architecture.json`, `configs/selected.toml`, one test metrics report.

## Documentation Requirements
Explain hypotheses/results, what stayed constant, acceptance thresholds, baseline, confidence/delta, leakage checks, bottleneck diagnosis, dropped components and limitations.

## Definition of Done
- Tests and `make ci` pass.
- Every meaningful component accepted/rejected with comparable evidence.
- Final architecture/config frozen before one test run.
- Gate 4 PASS when selected config is reproducible and no known leakage remains; weak model quality may be reported honestly without pretending success.

## Do Not
- Не cherry-pick runs with changed population.
- Не repeatedly tune on test or benchmark submissions.
- Не keep all models by default.
- Не interpret tiny noisy delta as certain improvement.
- Не hide resource cost/slice regressions.

## Handoff to Next Stage
Final-fit agent получает complete immutable `configs/selected.toml`, selected component list/hyperparameters, test-only report, source artifact lineage and no unresolved architecture choice.
