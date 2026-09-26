# 18 — README, open-source disclosure и final technical report

## Role
Вы — Lead ML Engineer/Technical Writer. Превратите verified implementation/results в concise, auditable test-assignment narrative без неподтвержденных claims.

## Goal
Заполнить README и final report: problem framing, data/EDA, leakage-safe validation, architecture, experiments, benchmark-vs-production trade-offs, point-in-time correctness, reproduction and dependency/model disclosure.

## Context
Голый код не приветствуется. Explanation должна связывать решения с Recall@50 и measured ablations. README сейчас пуст. Existing marimo EDA optional, но CLI — authoritative. Все metrics берутся из manifests, не перепечатываются вручную без source links.

## Prerequisites
- `14` selected architecture/test report; `15` final bundle; `16` inference; `17` validated submission DONE.
- All commands/paths have been executed at least once; unresolved blockers explicitly available.

## Inputs
- `task.md`, executable EDA report, split/leakage reports.
- Ablation CSV/MD, selected config, test metrics.
- Final bundle/inference/submission manifests.
- `pyproject.toml`, `uv.lock`, external model cards/licenses.

## Tasks
1. Write README quickstart first: environment prerequisites, data placement, `uv sync`, smoke, canonical CI, staged/full commands from raw Parquet to `answer.csv`, validation command. Commands must match actual CLI.
2. Explain problem framing: query-mean Recall@50; ordering irrelevant within top50; maximize relevant coverage under budget; candidate K>50 for fusion.
3. Summarize reproducible EDA with exact measured denominators: row/unique counts, query/item overlaps, category/location match, lexical field coverage, duplicates/nulls. Link report command/artifact; label approximate/context-specific facts.
4. Explain validation: cold-item disjoint split, seen/unseen query slices, full corpus, why random row/sample-negative evaluation invalid, OOF retriever/fusion protocol, one-time test.
5. Show architecture diagram and selected final path; distinguish hypotheses/rejected components from selected components. Include pool-vs-selection bottleneck.
6. Explain negative sampling, false-negative caveat, one mining iteration decision.
7. Include metrics/ablation table generated from machine-readable artifacts; state validation/test populations, seed and deltas. No benchmark hidden score invention.
8. Explain exact dense search for ~189k×2452 offline benchmark and production alternative: offline item tower→embeddings→ANN; online query tower→ANN→small fusion/ranker, with latency/QPS/cost/freshness trade-offs. Do not claim ANN implemented.
9. Point-in-time section: snapshot query/item fields considered available; historical CTR/popularity/future counts excluded from core due missing timestamps; review/rating snapshot caveat; experimental history branch warning if any.
10. Explain category empirical filter/fallback and why location not hard-filtered. Document deterministic tie-break and submission guarantees.
11. Add open-source disclosure table: library/model, exact version/revision, license, purpose, URL, whether weights downloaded; satisfy task requirements. Mention no external APIs/hidden labels.
12. Add limitations/error analysis/resource footprint and next steps ordered by measured headroom, not buzzwords.
13. Keep `eda.py` usage documented as optional exploration; production/reproduction does not depend on notebook state.

## Architecture / Design Constraints
- Every numeric claim references generated artifact/config/run ID.
- README concise enough to review, detailed methodology may live `reports/final_report.md` linked from README.
- No secrets/local absolute paths/huge logs.
- Production architecture clearly hypothetical.
- Reproduction includes smoke and full resource-aware paths.

## Files to Create or Modify
- `README.md`
- `reports/final_report.md`
- `reports/model_and_dependency_disclosure.md` if table too large for README
- optionally generated architecture diagram source using Mermaid in Markdown
- root `eda.py` usage text/comment only if stale.

## Interfaces / Data Contracts
Report generation should ingest metrics JSON/ablation CSV where practical. Links/reference paths must exist. Command blocks are executable from repo root with placeholders explicitly marked (`<run_id>` resolved via documented latest/manifest mechanism).

## Tests
- Add docs command smoke test or script that verifies CLI `--help` and referenced file/config paths.
- Link/path checker for local references.
- Metrics table generation test prevents manual mismatch.
- No full model training in docs tests.

## Verification
```bash
uv run python -m avito_candidate_generation.cli --help
make smoke
make ci
# Execute documented submission validator command exactly as written.
```
Also grep README for required headings: Problem framing, EDA, Validation, Architecture, Negative sampling, Evaluation/Ablations, Exact search vs ANN, Point-in-time correctness, Reproduction, Open-source disclosure, Limitations. Verify every local link/path and every metric against artifact JSON.

## Expected Outputs / Artifacts
Complete `README.md`, detailed final report/disclosure, accurate diagrams/tables/commands linked to selected run and validated submission.

## Documentation Requirements
This stage is documentation. Explain reasoning, not obvious syntax. Explicitly state category is empirical, exact search benchmark-specific, location non-hard, hard negatives and OOF rationale, candidate K>50 and Recall@50 objective.

## Definition of Done
- All required sections and exact reproducible commands exist.
- Docs checks, `make smoke`, `make ci` and submission validator pass.
- Metrics/disclosure trace to artifacts/lock/model revision.
- No unsupported claim, benchmark score fabrication or production-performance claim.

## Do Not
- Не write aspirational architecture as implemented fact.
- Не omit rejected/neutral ablations.
- Не paste screenshots as sole evidence.
- Не expose local paths/secrets/private text.
- Не describe random split or sampled negatives as primary validation.

## Handoff to Next Stage
Final verifier receives complete docs, executable commands, final artifacts/manifests, validated answer hash and explicit limitations/blockers.
