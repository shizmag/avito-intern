# Real neural validation status

- Primary split: clean cold-item validation.
- Dense: **PASS**, official artifact `artifacts/real_validation/dense_validation.json`.
- Two-Tower v1: **PASS**, official artifact `artifacts/real_validation/two_tower_v1_validation.json`.
- Two-Tower v2 hard-negative: **NOT_RUN**, artifact `artifacts/real_validation/two_tower_v2_validation.json` records no metrics.

## Resource protocol

- Device: CPU, selected for stable execution on current Mac.
- Generic item embeddings: float32 memmap, shape `[34370, 384]`.
- Two-Tower item embeddings: float32 memmap, shape `[34370, 32]`.
- Exact search: query batches of 128, item batches of 2048; no full query-item similarity matrix.
- Dense peak RSS: 3,024.7 MB measured by `resource.getrusage`.
- Two-Tower v1 peak RSS: 9,536.2 MB measured by `resource.getrusage`; this includes full CPU training process. Search itself used same chunked exact path.

## Metrics

| Method | R@10 | R@20 | R@50 | R@100 | R@200 | R@500 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.061533 | 0.095578 | 0.163485 | 0.230900 | 0.310752 | 0.437451 |
| Two-Tower v1 | 0.064044 | 0.114154 | 0.233249 | 0.382814 | 0.564581 | 0.775725 |

BM25 fixed baseline remains in `artifacts/metrics/real_validation.json`: R@50 0.399953, R@500 0.789121. Dense is complementary in principle but union metrics are not promoted because an independent union rerun was stopped after duplicate dense encoding exceeded resource budget. No union/champion claim is made.


## Union / fusion status

- Candidate-union oracle recall: **NOT_RUN**. Dense and TT artifacts currently persist embeddings and aggregate metrics, not per-query candidate tables; no second expensive exact-search pass was promoted as partial.
- RRF and CatBoost: **NOT_RUN** on real validation. Champion selection remains open.
- Release remains **BLOCKED**.
