# Avito Candidate Generation

Гибридный candidate generator, оптимизированный по **real cold-item validation Recall@50**.

## Champion

```text
query
  ├── bm25s BM25 (top-200)
  ├── intfloat/multilingual-e5-base, exact cosine search (top-200)
  └── task-specific Two-Tower v1 (top-200)
          ↓
  CatBoost selector
          ↓
  deterministic top-50
          ↓
  answer.csv
```

Выбранная конфигурация: [`configs/selected.toml`](configs/selected.toml).

- BM25 backend: `bm25s==0.2.14`, Robertson formulation;
- dense: `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`;
- E5 prefixes: `query: ` / `passage: `;
- dense search: exact, без ANN/vector DB;
- Two-Tower: v1 checkpoint; hard-negative v2 измерен, но исключён из champion из-за сильной регрессии;
- fusion: CatBoost, выбран по disjoint held-out Recall@50;
- category policy: `none` — см. ограничение validation ниже.

## Real validation

Split: deterministic cold-item validation, 43,413 queries, 34,370 items, 46,360 positive pairs. Primary metric — query-mean Recall@50.

### Category sanity check

- containment recall: **0.999849** (46,353 / 46,360 positive pairs);
- searchable items/query: mean **34,365.42**, median **34,367**, p95 **34,367**;
- validation category distribution is collapsed: 43,411 / 43,413 queries and 34,367 / 34,370 items have category `114`;
- global and pre-retrieval category-filtered BM25 therefore have identical R@50 (**0.400648**).

Итог: `category_policy = "none"`. Hard filtering корректно реализован до retrieval, но supplied validation не позволяет честно доказать его пользу из-за почти константной category.

### Retriever ablation

| Retriever | R@50 | R@200 | R@500 |
|---|---:|---:|---:|
| bm25s BM25 | 0.400648 | 0.653759 | 0.789355 |
| MiniLM dense baseline | 0.163485 | 0.310752 | 0.437451 |
| multilingual-e5-base | **0.441962** | **0.691567** | **0.818169** |
| Two-Tower v1 | 0.233249 | 0.564581 | 0.775725 |
| Two-Tower v2 hard-negative | 0.016511 | 0.048362 | 0.104670 |

TT v2 mined 293,022 leakage-safe BM25 negatives on train only; quality collapsed, so model is an ablation artifact, not selected component.

### True oracle candidate-pool coverage

`union@K` means full union of top-K **per source**; pool is not truncated back to K.

| Pool | Oracle R@50 | Oracle R@200 | Oracle R@500 |
|---|---:|---:|---:|
| BM25 ∪ E5 | 0.593522 | 0.834357 | 0.926154 |
| BM25 ∪ TT v1 | 0.503002 | 0.792455 | 0.908846 |
| E5 ∪ TT v1 | 0.545293 | 0.817313 | 0.923452 |
| BM25 ∪ E5 ∪ TT v1 | **0.653883** | **0.885474** | **0.957110** |

### Complementarity

Positive pairs recovered beyond BM25:

| Recovery | K=50 | K=200 | K=500 |
|---|---:|---:|---:|
| BM25 misses recovered by E5 | 8,847 | 8,404 | 6,338 |
| BM25 misses recovered by TT v1 | 4,776 | 6,529 | 5,609 |
| BM25+TT misses recovered by E5 | 6,906 | 4,305 | 2,198 |

Single-positive query quadrants at K=500 (41,401 queries):

- BM25 hit / E5 hit: 28,161;
- BM25 hit / E5 miss: 4,494;
- **BM25 miss / E5 hit: 5,680**;
- BM25 miss / E5 miss: 3,066.

### Fusion selection

RRF small grid selected `rrf_k=60`. On its held-out validation subset RRF R@50 = **0.497621**.

CatBoost was justified by the large oracle gap. Leak-safe protocol: deterministic 20% query split trains selector; disjoint 80% evaluates it. Paired comparison on the same 34,629 held-out queries:

| Fusion | Recall@50 |
|---|---:|
| RRF-60 | 0.489331 |
| CatBoost | **0.506448** |

Paired delta: **+0.017117**, bootstrap 95% CI **[+0.014231, +0.020074]**. CatBoost is selected.

## Artifacts

Key machine-readable evidence (generated, gitignored):

- `artifacts/real_validation/category_policy_validation.json`
- `artifacts/real_validation/dense_e5_base_validation.json`
- `artifacts/real_validation/two_tower_v1_validation.json`
- `artifacts/real_validation/two_tower_v2_validation.json`
- `artifacts/real_validation/union_validation_v2.json`
- `artifacts/real_validation/rrf_selection.json`
- `artifacts/real_validation/catboost_validation.json`
- `artifacts/real_validation/fusion_paired_comparison.json`
- `artifacts/selected/manifest.json`
- `artifacts/selected/predictions.parquet`
- `answer.csv`

Every experiment artifact records split/model/data fingerprints where applicable. `answer.csv` contains exactly 50 unique known items for each of 2,452 benchmark queries.

## Commands

```bash
uv sync --extra dev
make ci
.venv/bin/python -m avito_candidate_generation.verify --scope release
```

Selected workflow:

```bash
.venv/bin/python -m avito_candidate_generation.cli train-selected \
  --config configs/selected.toml
.venv/bin/python -m avito_candidate_generation.cli predict-selected \
  --manifest artifacts/selected/manifest.json \
  --output answer.csv
```

## Reproducibility and contracts

- seed: 42;
- stable item-ID tie-breaking;
- no train/validation item overlap;
- exact dense search in bounded query batches;
- BM25 returns unique known IDs and supports deterministic save/load;
- hard category filtering, when enabled, happens before retrieval;
- oracle union coverage is separate from fused Recall@K;
- train-only hard-negative mining removes all known positives;
- selector validation is disjoint from selector training;
- `answer.csv` is independently round-trip validated.

## Known data limitation

The supplied validation corpus is highly category-collapsed (`114`). Category containment is high, but this split cannot establish the expected speed/quality benefit of category partitioning on a naturally diverse category distribution. The selected global policy is therefore evidence-based for this validation set, not a universal product recommendation.
