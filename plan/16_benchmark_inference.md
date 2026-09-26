# 16 — Automated benchmark inference

## Role
Вы — ML Inference Engineer. Запустите immutable final bundle на benchmark query/item snapshot и создайте raw ranked predictions без ручных правок.

## Goal
Автоматизировать schema validation → preprocessing → selected retrievers → union/features/RRF/fusion → top-50 predictions с batching/cache/resource logs.

## Context
Benchmark: 2452 queries, 189212 items; all hidden relevant items are in corpus. Exact dense search preferred. Category policy selected validation stage; location remains feature, not unconditional filter. Output этого этапа — typed predictions artifact; CSV formatting/strict round-trip validation выполняет stage `17`.

## Prerequisites
- `15` verified final bundle DONE.
- Raw benchmark files validate against stage `02` contracts.
- No benchmark label access mechanism exists.

## Inputs
- `data/benchmark_queries.parquet`, `data/benchmark_items.parquet`.
- Immutable `artifacts/final/<run_id>/` bundle.
- `configs/final.toml` resource batch/device options.

## Tasks
1. Validate schemas/counts/unique IDs and bind raw fingerprints before inference. Preserve 16-char string IDs/case.
2. Preprocess benchmark queries/items with bundle's exact preprocessing config; cache by input/config hash.
3. Build/load selected BM25 index and generic/two-tower item embeddings for benchmark corpus. Compute each once; stale/corrupt cache rejected.
4. Run each selected retriever at frozen K and category global/hard/fallback policy. Exact dense dot/cosine in query/item batches/chunks; never full matrix.
5. Validate each candidate source; union without loss; compute identical pair features and RRF. If CatBoost selected, load bundle and predict; otherwise rank RRF. No dynamic fallback to unselected model.
6. Stable final sorting: model score desc, RRF desc, item_id ascending; select at most 50 unique known items/query. Queries with no selected candidates get documented empty list, though selected fallback should minimize this.
7. Save pre-top50 union features (or bounded selected diagnostics), final ranked Parquet, per-source counts/overlap, candidate length distribution, zero-candidate queries, timings/memory/cache hits.
8. Preserve all 2452 query IDs exactly once in final prediction representation. No manual editing.
9. Add resume semantics per completed source artifact; final manifest PASS only after all queries complete and feature schema matches bundle.
10. CLI command uses explicit bundle/path/output; no hardcoded run ID.

## Architecture / Design Constraints
- Offline exact search; ANN absent from benchmark execution.
- Streaming/batched; no giant Q×I matrix.
- Inference cannot import/use train labels except model artifacts already fit.
- Semantics frozen; only batch size/device/thread settings mutable.
- Deterministic result order across batch sizes/devices within declared numerical tolerance; compare top50, investigate tie changes.

## Files to Create or Modify
- `src/avito_candidate_generation/inference.py`
- inference section in `configs/final.toml`
- `tests/test_inference.py`
- `tests/integration/test_benchmark_smoke.py`

## Interfaces / Data Contracts
Final ranked Parquet:
```text
query_id:string
item_id:string
rank:int32 (1..<=50 contiguous/query)
final_score:float64 finite
rrf_score:float64 finite
```
Every benchmark query represented via separate query coverage table even if zero candidates. Manifest references bundle and raw fingerprints.

## Tests
- Tiny benchmark all queries included, known unique items only, top50/rank/tie rules.
- Category fallback for missing/small category; location mismatch retained.
- Batch/chunk sizes produce same rankings on deterministic fake encoders.
- Cache invalidates on corpus/bundle change.
- No train label dependency during fresh inference process.
- Corrupt/missing model/index fails instead of partial answer.

## Verification
```bash
uv run pytest tests/test_inference.py tests/integration/test_benchmark_smoke.py -q
uv run python -m avito_candidate_generation.inference predict --config configs/final.toml --bundle artifacts/final/<run_id> --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet
make ci
```
Assertions: 2452 distinct query IDs in coverage; <=50 unique known item IDs/query; ranks contiguous; IDs length 16 strings; finite scores; selected sources all complete; exact-search logs show bounded batch shapes, not full matrix allocation.

## Expected Outputs / Artifacts
`artifacts/inference/<run_id>/predictions.parquet`, optional candidate feature diagnostics, inference report/manifest/config, per-source cache references.

## Documentation Requirements
Explain execution order, exact-search batching, cache keys, category fallback, deterministic tie-break and why no ANN/manual edits.

## Definition of Done
- Tests and `make ci` pass.
- Full benchmark inference manifest PASS and all queries covered.
- Predictions meet typed top50/known-item invariants.
- No hidden labels/API/manual intervention used.

## Do Not
- Не format final CSV yet outside stage `17` canonical exporter.
- Не alter frozen K/model/features after seeing outputs.
- Не use train item lookup to replace retrieval.
- Не silently skip failed queries/source batches.
- Не implement ANN for this path.

## Handoff to Next Stage
Submission agent получает complete typed predictions Parquet, benchmark query/item tables, coverage/inference report and frozen bundle lineage.
