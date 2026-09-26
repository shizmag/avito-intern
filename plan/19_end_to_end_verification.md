# 19 — Final end-to-end verification and completion manifest

## Role
Вы — Independent ML Release Auditor. Не добавляйте модели; проверьте весь contract из clean reproducible state и выдайте machine-readable PASS/FAIL.

## Goal
Доказать путь raw parquet → validated data → leakage-safe split → selected models → benchmark predictions → round-trip-valid `answer.csv`; проверить tests/types/lint/coverage/determinism/docs/artifact lineage.

## Context
`code written != task completed`. Build alone недостаточен. Gate 5 требует final E2E, no known leakage, valid artifacts, truthful docs. Heavy full rerun может занимать часы; если resource blocker не позволяет, stage остаётся BLOCKED/FAIL, а максимально возможные checks всё равно выполняются.

## Prerequisites
- `01`–`18` заявлены DONE, all manifests/selected config/docs доступны.
- Working tree reviewed; raw data and required model caches/compute present or blocker recorded.

## Inputs
- Entire repository at candidate revision.
- Raw three parquet files.
- `configs/smoke.toml`, `configs/selected.toml`, `configs/final.toml`.
- Existing artifacts only as comparison; verifier должен отличать cache reuse от clean rebuild.

## Tasks
1. Audit every stage manifest and contract chain: input/output hashes, schema versions, selected config hash, no stale/mixed corpus/split/preprocessing artifacts.
2. Run clean environment sync and canonical `make ci`: format, lint, typecheck, unit/integration tests, coverage thresholds. Record exact counts/coverage, no invented summary.
3. Run deterministic smoke E2E twice in separate artifact dirs from tiny raw fixture: validate→canonicalize→preprocess→split→retrieve/train smoke→union/RRF/fusion as selected→infer→CSV→reread validate. Compare semantic fingerprints/top50.
4. Run or verify full data correctness gates and final bundle. For full heavy training, canonical reproducibility may use validated cached artifacts if hashes match; run final benchmark inference again with different batch size to test equivalence where feasible.
5. Re-run leakage audits: disjoint split items, no duplicate pairs, OOF heldout exclusions, no positive-as-negative, no positive injection, no benchmark labels, no history/target features, no accidental test tuning manifests.
6. Mechanically inspect public CLI/config entries and selected components. Ensure README commands and local paths work; dependency/model disclosure complete.
7. Re-run submission validator on released `answer.csv`; compare SHA with release manifest. Check all required format assertions.
8. Check artifact inventory, model/index loadability, no large generated artifacts tracked by git, no secrets, no absolute developer paths.
9. Produce `completion_manifest.json` with each acceptance predicate `PASS|FAIL|BLOCKED|N/A`, evidence command/artifact, timestamp/revision. Overall PASS only if every applicable required predicate PASS; unknown is not PASS.
10. Generate concise final audit report with metrics, commands, runtime/resources, remaining risks. Do not patch major issues silently; route root-cause fixes back to owning stage and rerun affected downstream checks.

## Architecture / Design Constraints
- Audit independent; no metric tuning or architecture changes.
- Cached full artifacts accepted only with verified hashes and load/inference probe.
- Smoke determinism expected exact semantically; GPU numerical variation uses previously declared tolerance and same ranking requirement where applicable.
- Fail early but continue independent checks to collect bounded diagnostics.
- Completion manifest authoritative.

## Files to Create or Modify
- `src/avito_candidate_generation/verify.py` or minimal verifier if not already present
- `tests/integration/test_end_to_end.py`
- generated `artifacts/verification/<run_id>/completion_manifest.json`
- `reports/final_verification.md`
- No model/config semantics changes.

## Interfaces / Data Contracts
Completion manifest minimum:
```json
{
  "stage":"final_verification",
  "status":"PASS|FAIL|BLOCKED",
  "git_commit":"...",
  "checks":{"ci":{"status":"PASS","evidence":"..."}},
  "metrics":{},
  "artifacts":{},
  "remaining_risks":[]
}
```
Required checks: outputs, schemas, tests, lint, types, coverage, determinism, leakage, metrics/selection provenance, bundle load, submission, docs, no secret/large tracked artifacts.

## Tests
- E2E tiny synthetic multi-positive/cold-item fixture.
- Verifier returns nonzero and manifest FAIL for one deliberately broken artifact/hash/submission/leak invariant.
- Two smoke runs semantic equality.
- Completion aggregation: BLOCKED/unknown cannot produce PASS.

## Verification
```bash
rm -rf /tmp/avito-smoke-a /tmp/avito-smoke-b
uv sync --all-groups
make ci
uv run python -m avito_candidate_generation.verify smoke --config configs/smoke.toml --output /tmp/avito-smoke-a
uv run python -m avito_candidate_generation.verify smoke --config configs/smoke.toml --output /tmp/avito-smoke-b
uv run python -m avito_candidate_generation.verify compare /tmp/avito-smoke-a /tmp/avito-smoke-b
uv run python -m avito_candidate_generation.pipeline verify-bundle --bundle artifacts/final/<run_id>
uv run python -m avito_candidate_generation.submission validate --submission answer.csv --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet
uv run python -m avito_candidate_generation.verify final --config configs/final.toml
```
Expected all exit 0 for DONE. Assert coverage >=85% line and >=75% branch, changed production >=90% when tooling supports it; answer checks all PASS; smoke fingerprints/top50 equal; final manifest overall PASS.

## Expected Outputs / Artifacts
Final completion manifest, audit report, CI/coverage evidence, deterministic comparison, leakage audit, bundle/submission verification and released answer hash.

## Documentation Requirements
Report exact commands/results, final Recall metrics with split/population (never hidden benchmark score unless externally supplied), selected architecture, artifacts, resource caveats and every non-PASS item. Distinguish implemented vs verified.

## Definition of Done
- Every applicable strict DoD predicate PASS; none unknown/BLOCKED.
- `make ci`, smoke twice, bundle verifier, submission validator and final verifier exit 0.
- Gate 5 PASS; completion manifest status PASS.
- If compute/data prevents full required check, report implementation/partial verification but stage is not DONE.

## Do Not
- Не fix failing test by weakening contract/coverage.
- Не claim likely pass for unexecuted command.
- Не regenerate selected architecture based on final/test/submission result.
- Не accept stale artifact because filename matches.
- Не mark BLOCKED/unknown check PASS.

## Handoff to Next Stage
Нет downstream implementation stage. Deliver `completion_manifest.json`, `reports/final_verification.md`, validated `answer.csv` hash and explicit remaining issues. Любая последующая change invalidates affected hashes and требует rerun verifier.
