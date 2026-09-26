# 08 — Supervised two-tower baseline

## Role
Вы — Deep Retrieval ML Engineer. Реализуйте task-specific two-tower, baseline training с in-batch negatives и leakage-safe validation.

## Goal
Обучить query/item encoders на train positives, сохранив independently precomputable item embeddings; сравнить baseline с generic dense/BM25 до hard-negative mining.

## Context
Train содержит positives без explicit negatives. Stage A использует in-batch negatives. Item tower обязан работать отдельно для offline embedding/production ANN concept. Category/location можно использовать как retrieval policy/fusion features; baseline encoder начинает с text only, чтобы не дублировать nearly-hard category assumption и сохранить переносимость. Structured embeddings добавлять только отдельным ablation later.

## Prerequisites
- `01`–`07` DONE; Gate 2 PASS and generic exact-search primitive ready.
- Cold-item split and OOF folds immutable.
- Torch/model dependencies/revision compatible and pinned.

## Inputs
- Fold-train positive pairs, canonical query/item text.
- Full validation corpus and ground truth.
- `configs/two_tower.toml`: base encoder/revision, shared vs untied towers, pooling, max lengths, embedding dim/projection, temperature, batch size, LR, epochs/steps, seed, device, precision, checkpoint policy, K/category strategy.

## Tasks
1. Choose simplest viable architecture: shared pretrained multilingual text backbone for query/item with configurable distinct templates; optional small projection + L2 normalization. Default shared weights reduce parameters. Do not add structured embeddings in baseline.
2. Enforce item encoding only from item snapshot fields and query encoding only from query fields. No cross-attention/candidate-dependent item representation.
3. Train contrastive softmax loss with in-batch negatives. Handle multiple positives/false negatives: mask batch pairs known positive for same query/item relation rather than label them negatives; deterministic sampler should minimize duplicate item/query collisions.
4. Seed Python/NumPy/torch/dataloader workers; report nondeterministic ops/device caveat. Gradient accumulation, mixed precision, batch sizes configurable.
5. Train only primary train interactions; early stopping/model choice by validation Recall@50 from periodic exact retrieval over full corpus (or a declared cheap proxy only for frequency, followed by full validation checkpoint comparison). Never train on val/test positives.
6. Save portable checkpoint: model config, base revision, weights, tokenizer, preprocessing fingerprint, training history. Verify fresh-process load.
7. Precompute validation corpus embeddings and exact retrieve K=500 using stage `07`; category strategies consistent with prior selected policy.
8. Evaluate R@K/slices, training curves, finite loss/grad, throughput/memory. Compare BM25, generic dense, two-tower and untruncated union pool recall.
9. Add smoke CLI using tiny fixture/fake-small local encoder path so CI need not download large model.

## Architecture / Design Constraints
- Text-only baseline; structured category/location remain downstream features/policy unless a later ablation proves encoder inclusion useful.
- Item embeddings independently precomputable — hard invariant/test.
- In-batch negatives stage only; no mined negatives yet.
- Full-corpus validation is authoritative, not batch loss.
- Checkpoint path never includes raw secrets; large weights gitignored.
- Training config immutable in checkpoint manifest.

## Files to Create or Modify
- `src/avito_candidate_generation/retrievers/two_tower.py`
- `src/avito_candidate_generation/training/two_tower.py`
- `configs/two_tower.toml`
- ML dependencies + `uv.lock`
- `tests/retrievers/test_two_tower.py`
- `tests/training/test_two_tower_smoke.py`

## Interfaces / Data Contracts
```python
class TwoTowerModel:
    def encode_queries(query_batch) -> Tensor: ...
    def encode_items(item_batch) -> Tensor: ...
```
Both output `[batch, dim]`, finite, optionally unit norm. Training rows reference IDs and text artifacts; never inline hidden val labels. Candidate output canonical source `two_tower_inbatch`.

## Tests
- Tensor shapes/norms, finite contrastive loss and backward/optimizer step.
- Item embedding unchanged by query batch and no query fields accepted by item tower.
- Known-positive false-negative mask hand fixture.
- Seeded sampler deterministic.
- Tiny overfit/smoke: train→serialize→fresh load→same embeddings/predictions within tolerance.
- Validation interactions excluded from train dataset.
- Candidate rank/schema/ties through exact search.

## Verification
```bash
uv run pytest tests/retrievers/test_two_tower.py tests/training/test_two_tower_smoke.py -q
uv run python -m avito_candidate_generation.training.two_tower train --config configs/smoke.toml
uv run python -m avito_candidate_generation.training.two_tower train --config configs/two_tower.toml
uv run python -m avito_candidate_generation.retrievers.two_tower evaluate --config configs/two_tower.toml --split validation
make ci
```
Required evidence: smoke finite forward/loss/backward; checkpoint reload equivalence; full validation R@K/slices; train/val item overlap zero; BM25/dense/two-tower union pool R@500.

## Expected Outputs / Artifacts
`artifacts/models/two_tower/<run_id>/checkpoint/`, training history/metrics, validation item embeddings/candidates, model card/config/manifest.

## Documentation Requirements
Explain two-tower serving property, text-only baseline choice, in-batch false-negative handling, exact validation search and structured-feature deferral.

## Definition of Done
- Tests and `make ci` pass; train/serialize/load/infer smoke complete.
- Full validation metrics measured without item leakage.
- Item tower independent and artifact reproducible.
- Baseline checkpoint chosen by validation Recall@50; test untouched.

## Do Not
- Не use cross-encoder/LLM reranker.
- Не feed category/location from query into item tower.
- Не treat all same-query positives as negatives.
- Не select checkpoint by training loss only.
- Не mine hard negatives inside this baseline stage.

## Handoff to Next Stage
Hard-negative agent получает trained in-batch checkpoint, exact retriever, known-positive map and source candidates from BM25/generic dense/current two-tower.
