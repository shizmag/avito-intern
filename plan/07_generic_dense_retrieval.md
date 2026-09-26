# 07 — Generic pretrained dense retrieval with exact search

## Role
Вы — Semantic Retrieval Engineer. Добавьте локальный multilingual dense source и exact batched search без привязки pipeline к одной модели.

## Goal
Измерить incremental semantic coverage generic embeddings относительно BM25 на том же validation protocol; сохранить reusable item embeddings и exact candidate outputs.

## Context
Corpus ~344k train items for validation and ~189k benchmark items. Offline benchmark has ~2452 queries, поэтому batched matrix similarity is preferable ANN: quality first, no approximation loss. Model must support Russian, run locally, and be configurable. This stage does not train model.

## Prerequisites
- `01`–`06` DONE; Gate 2 PASS.
- Preprocessing, category strategy candidates, evaluator and BM25 metrics available.
- Environment compatibility with torch/transformers/sentence-transformers explicitly checked.

## Inputs
- Field-aware normalized query/item representations and scalar categories.
- Full validation corpus/queries/ground truth.
- `configs/dense.toml`: model ID, immutable revision/commit when available, trust_remote_code default false, pooling/prefix, max lengths, batch sizes, dtype/device, normalize flag, K, category strategy.

## Tasks
1. Select one strong open-source multilingual/Russian-capable embedding model with documented license, model card and local inference. Keep adapter generic through config, but do not build plugin framework.
2. Pin runtime dependencies/model revision. Download/cache is explicit CLI step; unit tests never download. Fail clearly in offline mode if cache absent.
3. Compose query/item text field-aware. Follow model-required prefixes exactly; query and item encoders can differ only by prefix/template, not hidden labels.
4. Encode item corpus once in batches; encode queries in batches. Persist embeddings, ordered ID mapping, dtype/shape/norm stats and fingerprint. Reuse only on exact input/model/config match.
5. Implement exact dot-product/cosine top-K in query batches. Never materialize full Q×I matrix; keep per-batch top-K. Allow CPU/GPU and configurable item chunking for RAM/VRAM. Normalize embeddings if cosine.
6. Evaluate global, selected BM25 category policy, and hard/fallback category alternatives only as bounded comparison. If per-category matrix search used, preserve exactness within declared corpus.
7. Stable tie-break by item_id after similarity; finite score/norm validation; empty texts remain encoded under explicit policy.
8. Export canonical candidates at K 500 and metrics at required K/slices. Compare overlap/complementarity with BM25: intersection, unique relevant hits, union pool recall.
9. Log embedding/query time, search time, peak memory estimate, embedding bytes; no latency acceptance unless configured.
10. Choose generic dense config for downstream based on validation R@50 and incremental BM25 union R@500, not standalone vanity score.

## Architecture / Design Constraints
- No external API; local model only.
- Exact search is benchmark default; ANN may be discussed later, not implemented here.
- Item embeddings independently precomputable and immutable.
- Model-specific details remain config/template functions; public candidate schema model-agnostic.
- Float16 storage allowed only after validation against float32 tolerance/ranking delta; default safe dtype documented.
- Device-independent deterministic ordering; numerical tolerance explicit.

## Files to Create or Modify
- `src/avito_candidate_generation/retrievers/dense.py`
- `src/avito_candidate_generation/retrievers/exact_search.py`
- `configs/dense.toml`
- relevant dependencies + `uv.lock`
- `tests/retrievers/test_exact_search.py`
- `tests/retrievers/test_dense.py`
- optional heavy marker integration test using tiny local/fake encoder.

## Interfaces / Data Contracts
Encoder boundary:
```python
class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray: ...
```
Embedding artifact: contiguous `[n_items, dim]`, ordered `item_ids` strings, metadata model/revision/pooling/normalize/dtype/text fingerprint. Search returns canonical candidate rows.

## Tests
- Exact top-K matches brute-force NumPy on tiny matrices, with ties and chunked item search.
- Cosine normalization, NaN/zero-vector policy, K>corpus.
- Fake encoder verifies batching, text templates and no download.
- Cache invalidates on model revision/text config/corpus change.
- ID mapping/order preserved after save/load.
- Category fallback and stable tie ordering.
- Optional float16-vs-float32 ranking agreement diagnostic, not loose hidden tolerance.

## Verification
```bash
uv run pytest tests/retrievers/test_exact_search.py tests/retrievers/test_dense.py -q
uv run python -m avito_candidate_generation.retrievers.dense encode-items --config configs/dense.toml
uv run python -m avito_candidate_generation.retrievers.dense evaluate --config configs/dense.toml --split validation
make ci
```
Assertions: no full similarity matrix artifact; embeddings finite and row count equals corpus; max 500 unique candidates/query; exact tiny result correct; second encode is cache hit; metrics comparable to BM25 fingerprint/population.

## Expected Outputs / Artifacts
`artifacts/retrieval/dense/<run_id>/{item_embeddings,item_ids,candidates_validation}.(npy|parquet)`, `metrics.json`, BM25 complementarity report, manifest/config/model disclosure.

## Documentation Requirements
Explain chosen model/license/revision, prefixes/pooling, exact-vs-ANN decision, batching/memory formula, cache key and BM25 complementarity.

## Definition of Done
- Unit/integration tests and `make ci` pass.
- Generic dense R@K and slice metrics measured on same validation protocol.
- Item embedding reload reproduces candidates within explicit tolerance/order.
- Incremental union coverage vs BM25 is quantified; selected config or rejection documented.

## Do Not
- Не call remote inference APIs.
- Не add FAISS/ANN for benchmark path.
- Не silently use latest unpinned model revision.
- Не use benchmark score to choose model.
- Не materialize full pairwise similarity matrix.

## Handoff to Next Stage
Two-tower/union stages получают exact-search primitive, embedding artifact contract, generic dense candidates/metrics and measured complementarity with BM25.
