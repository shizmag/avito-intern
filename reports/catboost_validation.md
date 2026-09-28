# CatBoost Fusion Validation

Status: **PASS**.

## Summary

CatBoost fusion is evaluated using a leakage-safe protocol:
- **Split**: Hash bucket 0 (20%, 8,784 queries) trains the selector; disjoint hash buckets 1–4 (80%, 34,629 queries) evaluate ranking quality.
- **Candidate Pool Ablation**:
  - CatBoost over RRF top-500: **Recall@50 = 0.506692** (mean candidates = 500.0, oracle coverage = 90.88%).
  - CatBoost over Full Generator Union: **Recall@50 = 0.505516** (mean candidates = 1067.95, oracle coverage = 95.62%).
  - Delta: **-0.001176 (-0.12 pp)**, paired bootstrap 95% CI: `[-0.003174, +0.000784]`.
- **Decision**: RRF top-500 $\to$ CatBoost is selected as the champion fusion pipeline.
- **Artifacts**:
  - `artifacts/real_validation/catboost_candidate_pool_ablation.json`
  - `artifacts/real_validation/catboost_validation.json`
  - `artifacts/real_validation/fusion_paired_comparison.json`
