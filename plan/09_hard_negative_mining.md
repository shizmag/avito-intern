# 09 — Reproducible hard-negative mining

## Role
Вы — Retrieval Training Data Engineer. Создайте воспроизводимый hard-negative dataset из нескольких retrievers без positive contamination.

## Goal
Собрать сложные negatives из same-category BM25, generic dense и current two-tower results; сохранить provenance и bounded size для Stage B training.

## Context
Random negatives слишком просты. Ценны same-category, lexically/semantically similar и same/similar-location candidates. Однако train содержит только observed positives: unclicked retrieved item не гарантированно irrelevant. Поэтому artifacts называются sampled negatives, а не ground-truth negatives; known positives исключаются строго. Достаточно одной mining iteration, если retraining не показывает явный прирост.

## Prerequisites
- `01`–`08` DONE.
- Baseline checkpoint and source candidates exist for each training fold/split policy.
- Canonical known-positive map includes all canonical train positives for each `internal_query_id`.

## Inputs
- Fold-train query/item positives.
- Saved BM25/generic dense/two-tower indexes/embeddings/checkpoint.
- Item/query categories, locations, optional coordinates.
- `configs/hard_negatives.toml`: per-source top depth, max negatives/query, quotas, dedup priority, seed, category/location composition.

## Tasks
1. Mine candidates only with retrievers trained/fitted under allowed fold scope. For baseline Stage B train negatives, source indexes may use item text for all corpus items, but supervised model checkpoints cannot have learned held-out positive pairs.
2. Retrieve top results from same-category BM25 raw, generic dense and current two-tower. Optional small random same-category backfill only when source pool insufficient; never random-only dataset.
3. Build comprehensive known-positive set per query across canonical interactions; remove every `(query,item)` known positive before and after source union. Also reject self/duplicate rows.
4. Preserve source provenance: one negative row can have multiple source flags/ranks/scores; deterministic dedup. Add `same_category`, `same_location`, geo distance when valid, but no future/history aggregates.
5. Sample bounded max per query using configurable source quotas and deterministic seeded policy. Prefer diverse sources/ranks; do not let BM25 dominate solely by order.
6. Log candidate→filtered→selected counts, positive collision count, source overlap, category/location composition, per-query quantiles and queries without enough negatives.
7. Persist Parquet sorted by query, sampling order, item. Include mining model/index/config fingerprints and iteration=1.
8. Add validator CLI that exits nonzero on any known-positive collision, unknown IDs, duplicate final row, missing provenance, limit violation or stale upstream fingerprint.
9. Optional second iteration remains config-disabled and may only be enabled after Stage `10` demonstrates first iteration gain and oracle headroom.

## Architecture / Design Constraints
- Negatives are weak labels; do not call them true nonrelevant.
- No benchmark queries/items in training negatives unless explicitly part of final inference corpus and no labels; default excludes benchmark datasets entirely.
- Same category is preferred hardness, but preserve configurable fraction of cross-category/global negatives only if experiment rationale.
- Max negatives/query controls artifact and training balance.
- Source scores are diagnostics; training may consume IDs/text, not retrieval score as target shortcut.

## Files to Create or Modify
- `src/avito_candidate_generation/training/hard_negatives.py`
- `configs/hard_negatives.toml`
- `tests/training/test_hard_negatives.py`
- tiny candidate/positive fixtures.

## Interfaces / Data Contracts
Negative schema:
```text
internal_query_id:string
item_id:string
iteration:int8
sampling_order:int32
sources:list<string> or deterministic booleans
<source>_rank:int32 nullable
<source>_score:float64 nullable
same_category:bool
same_location:bool nullable
geo_distance_km:float64 nullable
```
Unique `(internal_query_id,item_id)`, `label=0` may be added only in training view. Manifest references known-positive fingerprint.

## Tests
- Any known positive returned by one/multiple retrievers is removed.
- Positive-not-used-as-negative after dedup/backfill.
- Source overlap/provenance retained.
- Max/query and source quotas; insufficient pools handled deterministically.
- Same seed/input gives byte/semantic identical artifact; source order permutation does not alter result after canonicalization.
- Unknown item, stale fingerprint and NaN score rejected.
- Query with all candidates positive returns fewer/zero negatives safely.

## Verification
```bash
uv run pytest tests/training/test_hard_negatives.py -q
uv run python -m avito_candidate_generation.training.hard_negatives mine --config configs/hard_negatives.toml
uv run python -m avito_candidate_generation.training.hard_negatives validate --config configs/hard_negatives.toml
make ci
```
Assertions: zero known-positive collisions; unique rows; all IDs in canonical corpus; per-query count <= configured max; source/composition table and collision counts emitted; rerun fingerprint identical.

## Expected Outputs / Artifacts
`artifacts/training/hard_negatives/<run_id>/negatives.parquet`, `negative_stats.json`, source composition CSV/Markdown, config/manifest.

## Documentation Requirements
Explain weak-negative assumption, why random-only negatives are insufficient, positive exclusion scope, quotas and why one mining iteration is default.

## Definition of Done
- Tests, validator and `make ci` pass.
- Zero known-positive contamination objectively verified.
- Artifact bounded, reproducible and traceable to retriever versions.
- Source difficulty/composition report exists.

## Do Not
- Не label retrieved unknown items as certain nonrelevant in docs.
- Не exclude only current row positive; exclude all known query positives.
- Не mine using model trained on heldout positives for OOF use.
- Не create unbounded cartesian training table.
- Не iterate mining without measured Stage `10` gain.

## Handoff to Next Stage
Two-tower retraining получает validated sampled negatives with source provenance, stable ordering and upstream model/data fingerprints.
