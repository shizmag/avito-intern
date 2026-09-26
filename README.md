# Avito Candidate Generation

Typed, deterministic candidate-generation pipeline.

## Checks

`make ci` runs Ruff, pytest, and Pyright.

Data boundary validates raw Parquet IDs and schemas. Canonical evaluation uses full-corpus query-mean Recall@K. Item splits are deterministic cold-item splits; benchmark labels never enter training.

## CLI

`PYTHONPATH=src .venv/bin/python -m avito_candidate_generation.data validate --config configs/base.toml`

`PYTHONPATH=src .venv/bin/python -m avito_candidate_generation.submission validate --submission answer.csv --queries data/benchmark_queries.parquet --items data/benchmark_items.parquet`
