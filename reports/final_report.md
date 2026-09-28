# Final report

Status: **VERIFIED**.

## Completed

- Validation corpus audit: `reports/validation_corpus_audit.json`, `reports/validation_corpus_audit.md`.
- Category ablation: `reports/category_policy_validation.json`; selected policy is `none` (empirical finding on collapsed validation category distribution).
- Exact chunked dense search: memory-bounded cosine similarity with zero vector-DB dependency.
- Dense multilingual-E5 real validation: `artifacts/real_validation/dense_e5_base_validation.json` (Recall@50 = 0.441962).
- Two-Tower v1 real validation: `artifacts/real_validation/two_tower_v1_validation.json` (Recall@50 = 0.233249).
- Two-Tower v2 hard-negative ablation: `artifacts/real_validation/two_tower_v2_validation.json` (Recall@50 collapsed to 0.016511; excluded).
- Jina Multilingual Reranker v2 zero-shot cross-encoder ablation: `reports/reranker_validation.md` (Recall@50 collapsed to 0.084286; non-commercial CC-BY-NC-4.0 license; excluded).
- CatBoost candidate pool ablation (apples-to-apples): `artifacts/real_validation/catboost_candidate_pool_ablation.json` (RRF top-500 Recall@50 = 0.506692 vs Full Union Recall@50 = 0.505516; Delta = -0.001176; selected RRF top-500 candidate pool).
- Selected pipeline trained and frozen: `artifacts/selected/manifest.json`.
- Benchmark inference executed: top-50 unique items for 2,452 queries.
- Submission artifact generated and validated: `answer.csv` (SHA-256 verified, 100% PASS on independent round-trip validator).

## Metrics Summary

| Method / Stage | Status | Recall@10 | Recall@20 | Recall@50 | Recall@200 | Recall@500 |
|---|---|---:|---:|---:|---:|---:|
| BM25 (`bm25s`) | Baseline | 0.182624 | 0.262741 | 0.400648 | 0.653759 | 0.789355 |
| Dense (`multilingual-e5-base`) | Baseline | 0.203814 | 0.283120 | 0.441962 | 0.691567 | 0.818169 |
| Two-Tower v1 | Baseline | 0.064044 | 0.114154 | 0.233249 | 0.564581 | 0.775725 |
| Two-Tower v2 (Hard-negative) | Ablation (Excluded) | 0.003921 | 0.007542 | 0.016511 | 0.048362 | 0.104670 |
| RRF (k=60) | Fusion Baseline | 0.210333 | 0.326333 | 0.497621 | — | — |
| Jina Reranker v2 Standalone | Ablation (Excluded) | 0.016000 | 0.030000 | 0.084286 | — | — |
| CatBoost + Jina Hybrid | Ablation (Excluded) | 0.146185 | 0.244578 | 0.409639 | — | — |
| CatBoost over Full Union | Candidate Pool Ablation | 0.235020 | 0.334929 | 0.505516 | — | — |
| **CatBoost over RRF top-500** | **Selected Champion** | **0.236905** | **0.336181** | **0.506692** | — | — |

## Final Champion Architecture

```text
Query
  ├── bm25s BM25 (top-500)
  ├── intfloat/multilingual-e5-base, exact cosine search (top-500)
  └── task-specific Two-Tower v1 (top-500)
          │
          ▼
  RRF top-500 candidate pool
          │
          ▼
  CatBoost selector
          │
          ▼
  top-50 unique item_id
          │
          ▼
      answer.csv
```

## Release Artifacts

- Selected pipeline manifest: `artifacts/selected/manifest.json`
- Selector model: `artifacts/real_validation/catboost_selector.cbm`
- Predictions: `artifacts/selected/predictions.parquet`
- Submission: `answer.csv` (2,452 queries $\times$ 50 unique items; independent validation passed)
- Release verification: `uv run avito verify --scope release` $\to$ **PASS (VERIFIED)**.
