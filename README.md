# Avito Candidate Generation

Reproducible candidate-generation pipeline for Avito search.

## Pipeline

```text
validated Parquet
  -> deterministic cold-item split
  -> BM25 lexical retrieval
  -> local transformer dense retrieval (exact cosine search)
  -> trainable query/item two-tower (in-batch negatives)
  -> candidate union and reciprocal-rank fusion
  -> optional OOF CatBoost selector
  -> deterministic unique top-50 inference
```

Production/default components fail loudly when required dependencies or model artifacts are absent. There is no hash-vector fallback under the production `TwoTowerModel` name. Hashing is used only to assign text tokens to trainable embedding rows; model parameters are optimized by backpropagation.

## Models and dependencies

- `torch`: trainable two-tower and contrastive optimization.
- `transformers`: local Hugging Face encoder adapter; model loading is `local_files_only=True`.
- `catboost`: learned fusion only when selected by validation.
- NumPy/Pandas/PyArrow: typed tables, exact search, and artifacts.

Selected dense default is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions), configured in `configs/selected.toml`. Model files must already exist in the local Hugging Face cache; missing cache raises an explicit error.

## Reproducibility and artifacts

`configs/selected.toml` is the single selected policy. A selected bundle stores `manifest.json` with retriever names, retrieval depth, fusion, metric, config hash, git revision, and component metadata. Dense item vectors are batched and persisted as `embeddings.npy`, `item_ids.json`, and `metadata.json`. Two-tower checkpoints use JSON metadata plus `safetensors`; no pickle-based model load is used.

All stochastic training receives an explicit seed. Item vectors do not depend on query text and can be precomputed. Exact matrix search is intentional for benchmark-scale corpora (~189k items); larger corpora/high-QPS serving should replace this bounded search with a measured ANN index, not add ANN by default.

## Commands

Install locked dependencies:

```bash
uv sync --dev
```

Quality gate:

```bash
make ci
```

ML checks:

```bash
make test-ml
make coverage
```

Validate raw data:

```bash
PYTHONPATH=src .venv/bin/python -m avito_candidate_generation.data validate --config configs/base.toml
```

Repository verifier / thin CLI:

```bash
PYTHONPATH=src .venv/bin/python main.py verify
```

Verifier reports `PASS` only after selected artifacts exist. Fresh checkout currently reports `BLOCKED` with missing `artifacts/selected/manifest.json`; this is intentional and prevents confusing source presence with completed training.

Selected artifact creation requires running selected retrieval/training workflow with local model cache and validation data; no fake manifest is generated.

Submission validation:

```bash
PYTHONPATH=src .venv/bin/python -m avito_candidate_generation.submission validate --submission answer.csv --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet
```

## Validation integrity

Training positives and benchmark labels are kept separate. Item splits are deterministic and cold-item based. Evaluation uses query-mean Recall@K over the declared ground-truth population; unknown IDs, duplicate candidates, non-finite scores, and invalid submission rows fail validation.

Real validation metrics must be generated from selected artifacts/config and recorded with data and corpus fingerprints. This repository does not claim a metric merely because a component exists.
