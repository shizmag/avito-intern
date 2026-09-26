# 10 — Two-tower training with hard negatives

## Role
Вы — Deep Retrieval ML Engineer. Дообучите/обучите two-tower на positives + mined negatives и докажите эффект full-corpus validation.

## Goal
Улучшить Recall@50 и/or complementary union coverage без leakage; сравнить Stage A in-batch baseline и Stage B hard-negative model under identical evaluation.

## Context
Hard negatives полезны только если улучшают retrieval objective. Training loss improvement недостаточен. One iteration preferred. Hard-negative ratios, temperature and batching affect false negatives/overfitting and remain config-driven.

## Prerequisites
- `08` baseline checkpoint/metrics and `09` validated negatives DONE.
- Negative manifest has zero known-positive collision and matching data/split fingerprints.

## Inputs
- Positive training pairs.
- `negatives.parquet` with provenance.
- Baseline model/checkpoint and `configs/two_tower_hard.toml`.
- Full validation corpus/ground truth/evaluator.

## Tasks
1. Implement training batches with each positive and configurable N mined negatives plus in-batch negatives. Preserve false-negative masks across all candidates.
2. Compare initialize-from-baseline fine-tuning vs configured fresh start only if resources permit; default fine-tune smallest path. Freeze/unfreeze policy explicit.
3. Weight/source sampling modestly configurable; initial policy uniform/diverse negatives, no large tuning grid. Do not feed source rank/score into encoder loss unless separately justified.
4. Deterministic sampler cycles/reseeds by epoch; cap repeated easy negative exposure. Log source/rank hardness and used counts.
5. Validate periodically using exact full-corpus retrieval. Checkpoint selection primary Recall@50; secondary union R@500 with BM25/generic dense and slice regression constraints.
6. Compare baseline vs hard model on identical validation query/corpus fingerprints: absolute/relative deltas R@K, seen/unseen, candidate overlap, training/resource costs.
7. Run optional single second mining iteration only if configured acceptance says first iteration gain exceeds threshold and remaining oracle gap justifies it. Otherwise document N/A.
8. Save portable checkpoint and precomputed candidates; fresh-process reload equivalence.
9. Freeze selected two-tower retriever config for candidate union. A regression may still be accepted only if union oracle R@500 improves and later selection benefits, with explicit evidence.

## Architecture / Design Constraints
- Item tower independence unchanged.
- No validation positive enters train or mining model scope.
- Validation corpus full; no sampled-negative model selection.
- Same preprocessing/base model revision for fair baseline unless change is isolated and reported.
- Numerical/GPU determinism tolerance documented.

## Files to Create or Modify
- `src/avito_candidate_generation/training/two_tower.py`
- `configs/two_tower_hard.toml`
- `tests/training/test_two_tower_hard.py`
- update model comparison reporting utilities only if reused.

## Interfaces / Data Contracts
Training dataset joins positive query to positive item and N unique negatives from same `internal_query_id`. Any missing item/text is fatal. Candidate source `two_tower_hard`. Checkpoint manifest includes parent checkpoint and negative artifact hashes.

## Tests
- Batch construction labels/masks for multiple negatives and known multi-positives.
- No duplicate positive/negative IDs in example.
- Finite hard-negative loss/backward; N=0 reproduces baseline code path.
- Deterministic sampler.
- Tiny train→serialize→reload→exact retrieval.
- Split leak assertion before fit.

## Verification
```bash
uv run pytest tests/training/test_two_tower_hard.py tests/retrievers/test_two_tower.py -q
uv run python -m avito_candidate_generation.training.two_tower train --config configs/two_tower_hard.toml
uv run python -m avito_candidate_generation.retrievers.two_tower evaluate --config configs/two_tower_hard.toml --split validation
make ci
```
Report must show baseline and hard model R@10/20/50/100/200/500, slices, union pool recall, absolute delta, candidate overlap, time/memory. Assert train items/pairs comply with split and negative collisions remain zero.

## Expected Outputs / Artifacts
Hard-trained checkpoint, training history, validation embeddings/candidates, comparison JSON/CSV/MD and selected two-tower config/manifest.

## Documentation Requirements
Explain hard-negative batch objective, false-negative masking, model selection metric, one-iteration decision and any slice regression.

## Definition of Done
- Tests and `make ci` pass; checkpoint reload works.
- Full validation comparison on identical fingerprints complete.
- Selected baseline/hard model choice machine-readable and justified by Recall@50/union coverage.
- Test remains untouched.

## Do Not
- Не declare success from lower loss.
- Не reuse validation positives as negatives/train rows.
- Не run indefinite mining loops.
- Не change base model and negative scheme simultaneously without isolated ablation.
- Не hide regressed slices.

## Handoff to Next Stage
Union stage получает selected BM25, generic dense and selected two-tower candidates, complete per-source metrics, exact fingerprints and source nomenclature.
