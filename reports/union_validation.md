# Candidate union and RRF validation

Status: **PASS** for measured clean validation sources.

| Method | R@50 | R@100 | R@200 | R@500 |
|---|---:|---:|---:|---:|
| bm25 | 0.399999 | 0.524870 | 0.654113 | 0.789167 |
| dense | 0.163485 | 0.230900 | 0.310752 | 0.437451 |
| two_tower_v1 | 0.233249 | 0.382814 | 0.564581 | 0.775725 |
| union_bm25_dense | 0.399976 | 0.525561 | 0.656681 | 0.796349 |
| union_bm25_tt | 0.399976 | 0.525723 | 0.660467 | 0.808232 |
| union_dense_tt | 0.235150 | 0.383395 | 0.560595 | 0.768433 |
| union_all | 0.399976 | 0.525815 | 0.660708 | 0.808900 |
| rrf_v1 | 0.412826 | 0.553685 | 0.700724 | 0.853780 |

- RRF: `rrf_k=60`, source budgets `K=500`.
- Resource: CPU, query batch 128, item batch 2048, peak RSS 4470.7 MB, runtime 762.2 s.
- Diagnosis: mixed bottleneck; selection dominant. `union_all` R@500=0.808900 versus RRF R@50=0.412826.
- CatBoost: NOT_RUN.
- Champion: not selected; TT v2 and CatBoost gates remain open.
