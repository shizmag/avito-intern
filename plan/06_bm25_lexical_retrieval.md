# 06 — Strong BM25 lexical retrieval и category strategy

## Role
Вы — Information Retrieval Engineer. Постройте сильный, объяснимый lexical baseline и измерьте raw/lemma/field/category варианты.

## Goal
Получить воспроизводимые BM25 candidates по full corpus, узнать Recall@50/100/200/500, проверить category filtering и лемматизацию, сохранить индекс и baseline metrics. Это Gate 2 до neural training.

## Context
Positive lexical coverage велико, поэтому BM25 не dummy. Query использует `search_query` + params; item fields title, params, description различаются по сигналу. Category совпадает ~0.9999+, но это empirical property; location ~0.84 и не hard filter. Candidate K >50 нужен, чтобы fusion имел coverage.

## Prerequisites
- `01`–`05` DONE; Gate 1 PASS.
- Canonical field-aware text, validation corpus/queries/ground truth, evaluator доступны.

## Inputs
- Preprocessed query/item artifacts raw-normalized and optional lemma.
- Validation config; BM25 config with K `[50,100,200,500]`, default candidate export K=500.
- Category/location scalar fields.

## Tasks
1. Выбрать минимальную локальную BM25 implementation/dependency после проверки installed stack; pin dependency. Индекс должен serializable/loadable. Не писать fragile BM25 с нуля без нужды.
2. Реализовать unified retriever output. Query tokens: normalized query plus params, с configurable query field boost/repetition только если library API требует.
3. Baseline A: BM25 over explicit weighted/labeled concatenation. Baseline B: field-aware weighted score `w_title*BM25_title + w_params*BM25_params + w_description*BM25_description`; начать с малого manual set, не grid. Если field-aware implementation чрезмерна, сначала concatenated, затем добавить только при measured need.
4. Ablate raw normalized vs lemma. Raw обязателен; lemma optional и включается final только если validation R@50 или pool recall materially improves с указанным порогом/ресурсной ценой.
5. Compare category strategies on same queries/corpus/config: global; hard category; category-first + configurable global fallback/reserved pool. Missing/unknown/empty category всегда имеет deterministic global fallback.
6. Location не hard-filter. Можно report same-location share among returned candidates; ranking boost остаётся fusion stage.
7. Exact lexical retrieval top K per query; stable tie-break by item_id. Если category has fewer K, return all and fallback according to selected strategy.
8. Cache index keyed by corpus/text/config fingerprint. Load validates fingerprint; corrupt/stale index rejected.
9. Evaluate full validation and slices at all K; report build/query latency, index size, peak RSS estimate and candidate count.
10. Save experiment table and choose BM25 config for downstream by primary validation R@50, tie-break pool R@500/resource simplicity. Не читать test.
11. CLI build/retrieve/evaluate with smoke and full modes.

## Architecture / Design Constraints
- Unified interface may be Protocol/function; no abstract framework unless second implementation needs it.
- Field scores normalized only if validated; raw BM25 scores across separate field indexes may need explicit weighted sum, documented.
- Candidate table has no duplicate `(query, item, source)`; ranks 1-based contiguous.
- Source names encode variant (`bm25_raw`, `bm25_lemma`) but configs/manifests carry full details.
- Index persistent; no rebuild if fingerprint matches.
- Candidate retrieval full corpus/category corpus, not sampled negatives.

## Files to Create or Modify
- `src/avito_candidate_generation/retrievers/bm25.py`
- minimal shared candidate Protocol/schema module if not already present
- `configs/bm25.toml`
- dependency in `pyproject.toml`, regenerated `uv.lock`
- `tests/retrievers/test_bm25.py`
- `tests/integration/test_bm25_pipeline.py`

## Interfaces / Data Contracts
```python
def retrieve_bm25(queries, items, *, config) -> CandidateTable: ...
```
Output canonical candidate schema; finite score, rank `[1..n]`, max K. Index manifest contains corpus fingerprint, preprocessing version, variant, BM25 params/field weights/dependency version.

## Tests
- Tiny corpus expected ranking for title/query match.
- Null/empty query produces deterministic empty/fallback behavior, not crash.
- Category hard filter returns only matching category; fallback fills when category small/unknown.
- Location mismatch not removed.
- Stable ties/order; no duplicate candidate rows; rank contiguous.
- Raw vs lemma use distinct cache keys.
- Saved/reloaded index returns identical candidates.
- Integration tiny canonical→index→retrieve→evaluate.

## Verification
```bash
uv run pytest tests/retrievers/test_bm25.py tests/integration/test_bm25_pipeline.py -q
uv run python -m avito_candidate_generation.retrievers.bm25 build --config configs/bm25.toml
uv run python -m avito_candidate_generation.retrievers.bm25 evaluate --config configs/bm25.toml --split validation
make ci
```
Required report rows: raw concatenated, raw field-aware if implemented, lemma variant, and global/hard/fallback category strategies. Columns: R@50/100/200/500, seen/unseen R@50, pool size, latency, index size. Category hard/fallback decision includes delta and slice regression.

## Expected Outputs / Artifacts
`artifacts/retrieval/bm25/<run_id>/index/`, `candidates_validation.parquet`, `metrics.json`, `ablation.csv`, `ablation.md`, manifest/config.

## Documentation Requirements
Explain candidate K>50, field weighting rationale, lemma decision, category empirical status/fallback and why location is not filtered. Record exact dependency/license.

## Definition of Done
- Tests, typecheck/lint and `make ci` pass.
- Full validation BM25 metrics known; artifacts reproducible after index reload.
- Selected BM25 downstream config frozen with evidence.
- Gate 2 PASS: raw baseline, pool R@200/R@500 and category impact available. If no variant helps, raw global/fallback remains valid baseline and failure is documented.

## Do Not
- Не use labels to alter per-query retrieval.
- Не tune dozens of weights.
- Не hard-filter location.
- Не force category filter without slice evidence/fallback.
- Не discard raw variant because lemma sounds linguistically better.

## Handoff to Next Stage
Dense/two-tower/union stages получают validated BM25 candidate source, persistent index, selected category policy and comparable baseline metrics.
