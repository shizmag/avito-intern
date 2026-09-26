# 11 — Candidate union, RRF и pair-level features

## Role
Вы — Retrieval Fusion Feature Engineer. Объедините candidate sources без потери coverage, реализуйте RRF baseline и deterministic pair features.

## Goal
Создать единый deduplicated candidate pool, source score/rank features, RRF ranking и inference-safe pair features для OOF/CatBoost.

## Context
Если union pool Recall@500 высок, но final R@50 ниже, bottleneck — selection. RRF нужен как strong baseline и CatBoost feature. Он не должен destructive-prefilter candidates до CatBoost без measured evidence. Scores разных retrievers несопоставимы напрямую; ranks/RRF and normalized within-query features safer.

## Prerequisites
- `06`, `07`, `10` selected retriever candidates and metrics DONE.
- Canonical evaluator and schemas available; all candidates refer to same split/query/corpus fingerprints.

## Inputs
- Per-source canonical candidate Parquet, default K up to 500.
- Canonical query/item scalar features and safe text/filter match computations.
- `configs/fusion_features.toml`: enabled sources, per-source K, RRF k, missing rank convention, numeric cleaning policies.

## Tasks
1. Validate upstream compatibility; reject mixed ground truth/corpus/query/preprocessing fingerprints.
2. Outer union on `(internal_query_id,item_id)`, one row per pair. Preserve per-source raw score/rank, source indicator and count. Dedup source candidates before join; report overlap.
3. Calculate RRF `sum_s 1/(rrf_k + rank_s)` only for present source; `rrf_k` configurable. Rank by RRF desc, then best source rank, then item_id.
4. Export RRF top K and evaluate R@K/slices as selection baseline. Also calculate untruncated pool coverage at declared source depths.
5. Build inference-safe pair features: per-source score/rank/presence; within-query score z/range/percentile only with zero-variance safety; RRF; retriever count; category equality; same location; haversine distance only valid finite coords; normalized lexical overlap/filter-match; cleaned log1p reviews; rating missing/value; robust price log/missing; phone/message flags.
6. Category-match feature may be dropped when constant under hard filter, but schema/config records it. No historical popularity/CTR.
7. Numeric policy explicit: missing indicators, no arbitrary zero semantics; finite output enforced. Decimal price parse invalid/negative as missing according to data report.
8. Preserve all union candidates for downstream by default. Optional pool cap >50 may only be enabled after pool-recall ablation; never silently take RRF top-N before CatBoost.
9. Feature computation shared by OOF training and benchmark inference; add equivalence test.
10. Write feature dictionary with dtype, semantics, availability and leakage status.

## Architecture / Design Constraints
- Pair table row uniqueness hard invariant.
- No label columns in generic feature builder; label join happens in OOF stage.
- Retrieval scores not globally calibrated; transformations per query/source.
- Haversine handles missing/out-of-range coordinates and Earth units km.
- Feature set small and auditable; no target/history aggregate.
- Stable deterministic sorting/order.

## Files to Create or Modify
- `src/avito_candidate_generation/fusion/rrf.py`
- `src/avito_candidate_generation/fusion/features.py`
- `src/avito_candidate_generation/candidates.py`
- `configs/fusion_features.toml`
- `tests/fusion/test_rrf.py`
- `tests/fusion/test_features.py`
- `tests/test_candidates.py`

## Interfaces / Data Contracts
Wide pair schema:
```text
internal_query_id:string, item_id:string
<source>_present:bool
<source>_score:float nullable
<source>_rank:int nullable
rrf_score:float
retriever_count:int
...safe pair/item features...
```
Unique pair, no labels. Feature dictionary JSON version binds columns/dtypes/order.

## Tests
- Union dedup with overlapping sources; source provenance retained.
- RRF hand calculation, missing source, configurable k, stable ties.
- Untruncated union set recall monotonic vs included source pools under same depths.
- Null/invalid rating/price/coords, zero-variance score normalization, haversine known distance.
- Location mismatch retained.
- Feature builder same output via training/inference entrypoints.
- No nonfinite numeric features and no history/target field.

## Verification
```bash
uv run pytest tests/test_candidates.py tests/fusion/test_rrf.py tests/fusion/test_features.py -q
uv run python -m avito_candidate_generation.candidates union --config configs/fusion_features.toml --split validation
uv run python -m avito_candidate_generation.fusion.rrf evaluate --config configs/fusion_features.toml --split validation
make ci
```
Assertions: unique pairs; all source ranks/scores retained; pool count and pool recall by source/union; RRF R@K and slices; no candidate lost before optional explicitly measured cap; all model feature numerics finite after declared encoding.

## Expected Outputs / Artifacts
`artifacts/fusion/features/<run_id>/candidate_union_validation.parquet`, `rrf_predictions.parquet`, `feature_dictionary.json`, overlap/oracle/metrics reports, manifest/config.

## Documentation Requirements
Explain pool recall vs final selection, score comparability, RRF formula/k, non-destructive union, missing-value semantics and inference-safe feature rationale.

## Definition of Done
- Tests and `make ci` pass.
- Union and RRF metrics reproducible; overlap/oracle table complete.
- Pair feature schema versioned and finite; train/inference equivalence proven.
- No target/history leakage and no unexplained pool truncation.

## Do Not
- Не inner join sources.
- Не fill missing rank with 0/best rank.
- Не discard non-RRF top candidates before CatBoost by default.
- Не hard-filter location.
- Не add item popularity/CTR from train history to core features.

## Handoff to Next Stage
OOF agent получает tested union/feature builder, feature dictionary, source configs/checkpoints and RRF baseline metrics.
