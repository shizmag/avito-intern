# 05 — Canonical retrieval evaluation framework

## Role
Вы — ML Metrics Engineer. Реализуйте единственный canonical evaluator, используемый всеми retrievers, fusion experiments и final verification.

## Goal
Обеспечить mathematically correct query-mean Recall@K на realistic full corpus, slice metrics, contract validation и machine-readable comparable reports.

## Context
Primary objective: для каждого query вернуть до 50 items; порядок внутри top-50 не влияет. Для diagnostics нужны K = 10, 20, 50, 100, 200, 500. Multi-positive query denominator — число всех relevant items. Sampling negatives запрещён. Retriever outputs должны сравниваться только на одинаковых query population, corpus и ground truth fingerprint.

## Prerequisites
- `01`–`04` DONE, Gate 1 data/split artifacts готовы.
- Прочитаны split lifecycle и unified candidate schema из `00`.

## Inputs
- `ground_truth.parquet`, `query_slices.parquet`, split/corpus manifests.
- Candidate Parquet canonical schema.
- Config K list and evaluation split.

## Tasks
1. Реализовать `Recall@K = mean_q(|topK_q ∩ relevant_q| / |relevant_q|)` с dedup candidate items до cutoff и stable score/rank ordering.
2. Ground-truth population authoritative: queries без predictions получают recall 0; prediction-only/unknown queries fail contract; ground-truth query без relevant item fail input contract.
3. Validate candidates: IDs string, finite scores, ranks positive/contiguous per source where source evaluation; no unknown item IDs; duplicate query-item policy — fail canonical input или explicit deterministic dedup before metric with count reported.
4. Tie-break deterministic: primary score/rank, then `item_id` ascending. Не зависеть от dataframe order.
5. Считать overall K list и slices: seen/unseen query text; single/multi-positive; same/cross-location relevant population; with/without filters. Для query with mixed location positives определить documented query-level slice semantics либо positive-weighted secondary metric; не менять canonical mean silently.
6. Report numerator diagnostics, number of evaluated/all queries, excluded=0 by default, relevant item count distribution, prediction length, duplicate/unknown counts.
7. Добавить source/union comparison table API. Oracle union Recall@K = recall after set union of source top pools, then either pool-coverage at declared pool size or ranked topK under explicit rule. Не утверждать `union R@K >= source R@K`, если union был reranked/truncated; guarantee applies to untruncated set coverage or union pool with same included source candidates.
8. Добавить latency/resource metadata passthrough, но не смешивать его с quality metric.
9. CLI evaluate/compare writes JSON + CSV/Markdown table; nonzero exit for schema/invariant or configured acceptance threshold failure.

## Architecture / Design Constraints
- One implementation for all stages; no metric copies in notebooks/retrievers.
- Ground truth and predictions loaded by projected columns/batches where useful.
- Floating values in [0,1]; empty prediction list valid and scores 0.
- Test split report generation gated by explicit `--allow-test-evaluation` and architecture freeze metadata.
- Metrics report includes candidate/config/data fingerprints.

## Files to Create or Modify
- `src/avito_candidate_generation/evaluation.py`
- evaluation config in `configs/validation.toml`
- `tests/test_evaluation.py`
- `tests/test_candidate_contract.py`
- small golden ground-truth/prediction fixtures.

## Interfaces / Data Contracts
```python
def recall_at_k(
    predictions: DataFrameLike,
    ground_truth: DataFrameLike,
    k: int,
) -> float: ...
```
Metrics JSON: schema version, stage/run/split, corpus/ground-truth/candidate fingerprints, `metrics` keys `recall@10`…`recall@500`, slice objects with support and recall, counts, timings, status.

Candidate table schema exactly follows `00_overview.md`; source-specific evaluation requires one source per row.

## Tests
- Hand-calculated single/multi-positive queries.
- Empty predictions, fewer than K, K larger than corpus, duplicate predictions, ties, unknown IDs/queries, missing ground-truth query predictions.
- Query mean differs from pooled micro recall in fixture; assert canonical query mean.
- Input order permutation leaves metrics unchanged.
- Recall@K monotonic for a fixed ranking and K sequence.
- Untruncated union pool recall >= each included source pool recall under same pool definition.
- Slice supports and values hand-calculated.
- Test gate refuses accidental test evaluation.

## Verification
```bash
uv run pytest tests/test_evaluation.py tests/test_candidate_contract.py -q
uv run python -m avito_candidate_generation.evaluation evaluate --ground-truth tests/fixtures/ground_truth.parquet --candidates tests/fixtures/candidates.parquet --split validation --output /tmp/avito-metrics
make ci
```
Assertions: all metrics finite/in [0,1], exact golden values match, denominators/support printed, output JSON parses and status is PASS. Run twice: byte-stable metrics except explicitly volatile timestamps stored only in manifest.

## Expected Outputs / Artifacts
Canonical evaluator module/CLI; golden metrics; future `artifacts/metrics/<run_id>/metrics.json`, `metrics.csv`, `comparison.md`, manifest.

## Documentation Requirements
State formula, query averaging, duplicate and tie rules, missing prediction semantics, slice populations, and pool recall vs ranked Recall@K distinction.

## Definition of Done
- Metric edge/property tests and `make ci` pass.
- No retriever-specific evaluator remains.
- Evaluator rejects incompatible fingerprints/schema.
- Gate 1 complete: data, split, preprocessing and metrics contracts verified.

## Do Not
- Не evaluate only successful queries.
- Не drop unknown/difficult rows silently.
- Не use 1 positive + random negatives.
- Не tune on fixed test.
- Не claim union dominance for a truncated reranked list without correct definition.

## Handoff to Next Stage
BM25/dense/two-tower agents получают canonical candidate validator, evaluator CLI, fixed validation ground truth/corpus and comparable metrics schema.
