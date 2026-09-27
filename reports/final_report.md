# Final report

Status: **BLOCKED**.

## Completed

- Validation corpus audit: `reports/validation_corpus_audit.json`, `reports/validation_corpus_audit.md`.
- Category ablation: `reports/category_ablation.json`, `reports/category_ablation.md`; selected config records `category_policy = "hard"`.
- Exact chunked top-k correctness test: `tests/test_dense.py`.
- Dense real validation: `artifacts/real_validation/dense_validation.json`.
- Two-Tower v1 real validation: `artifacts/real_validation/two_tower_v1_validation.json`.

## Metrics

| Method | R@10 | R@20 | R@50 | R@100 | R@200 | R@500 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.182624 | 0.262741 | 0.399953 | 0.524824 | 0.654067 | 0.789121 |
| Dense | 0.061533 | 0.095578 | 0.163485 | 0.230900 | 0.310752 | 0.437451 |
| Two-Tower v1 | 0.064044 | 0.114154 | 0.233249 | 0.382814 | 0.564581 | 0.775725 |
| Two-Tower v2 hard-negative | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## Resource behavior

- Device: CPU.
- Dense item artifact: float32 memmap, `[34370, 384]`; query batch 128; item batch 2048; peak RSS 3,024.7 MB.
- Two-Tower v1 item artifact: float32 memmap, `[34370, 32]`; same exact chunked search; full training peak RSS 9,536.2 MB.
- No full query×item similarity matrix created.

## Remaining gates

Candidate union oracle recall, RRF, CatBoost comparison, champion selection, final full-data training, benchmark inference, answer.csv, and strict release verification remain incomplete. No champion is selected.

Machine-readable consolidated status: `reports/real_validation.json`.
