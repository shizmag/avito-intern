# 03 — Единый text preprocessing layer

## Role
Вы — NLP/IR Engineer для русского языка, отвечающий за deterministic representations без потери raw данных.

## Goal
Создать единый preprocessing API для query/item text: raw, normalized и optional lemmatized representations; обеспечить одинаковое поведение train/evaluation/benchmark.

## Context
Лексическое покрытие положительных пар высокое: title ~56%, title+params ~68%, с description ~85%. Поля имеют разную ценность, поэтому нельзя необратимо склеить их на ingestion. Лемматизация — эксперимент, а не обязательная истина. Null/empty/long text должны обрабатываться явно.

## Prerequisites
- `01` и `02` DONE; canonical data artifacts и schema доступны.
- Data fingerprint и internal query identity зафиксированы.

## Inputs
- Canonical query/item Parquet из `artifacts/data/...`.
- Config sections `[text]`, `[resources]`.
- Russian text in `search_query`, `search_infm_params_text`, `item_title_raw`, `item_infm_params_text`, `item_description_raw`.

## Tasks
1. Реализовать pure `normalize_text`: Unicode NFKC, lowercase, optional configurable `ё→е`, whitespace collapse, stable punctuation separation/removal policy. Не transliterate и не удалять digits по умолчанию.
2. Null text преобразовывать в empty representation только в preprocessing output, сохраняя raw null и missing indicator для downstream features.
3. Реализовать deterministic tokenizer для BM25: token regex/min length configurable; numbers and mixed brand/model tokens policy покрыть tests.
4. Реализовать lemmatization как отдельный opt-in representation с local open-source backend, предпочтительно `pymorphy3` при подтвержденной совместимости. Cache lemma by unique token. Если backend недоступен, raw pipeline работает, lemma command fail/skip только явно по config и с report; silent fallback запрещён.
5. Сформировать field-aware structures: query text = отдельно query и params; item = отдельно title, params, description. Дополнительно дать deterministic labeled concatenation для dense encoders (`query: ...`, `params: ...`, `title: ...`, etc.), сохраняя модельные prefix requirements configurable.
6. Ограничения длины разделить: lexical token cap и dense tokenizer max length не должны быть одной константой. Truncation policy и rates логируются по полям; title/params приоритетнее tail description.
7. Добавить batch/materialize CLI с cache key = input fingerprint + preprocessing config/version. Output sorted by ID and supports resume only после manifest validation.
8. Измерить на full data: null/empty rates, token-length percentiles, truncation rates, unique token count, lemma runtime. Это diagnostics, не quality claim.

## Architecture / Design Constraints
- Raw fields никогда не overwrite.
- Один implementation используется retrieval/training/inference; training-serving skew test обязателен.
- Normalization не должна менять IDs или scalar category/location.
- Avoid fit on labels. Vocabulary/IDF строится позже retriever на corpus.
- Field weights не применять здесь; это BM25 config.
- No external API/network inference.
- Batch processing and on-disk cache; не создавать Python-object copy полного корпуса без нужды.

## Files to Create or Modify
- `src/avito_candidate_generation/preprocessing.py`
- config sections in `configs/base.toml`, `configs/smoke.toml`
- optional dependency and `uv.lock` only for selected local lemmatizer
- `tests/test_preprocessing.py`
- `tests/fixtures/text_cases.json` or equivalent human-readable fixture

## Interfaces / Data Contracts
```python
@dataclass(frozen=True)
class TextRepresentations:
    raw: str | None
    normalized: str
    tokens: tuple[str, ...]
    lemmatized: str | None
    was_missing: bool
    was_truncated: bool
```
Batch outputs use IDs plus per-field columns such as `<field>_normalized`, `<field>_lemma`, `<field>_missing`; exact schema versioned. Dense composed text function accepts explicit mode/model prefix config.

## Tests
- Cyrillic, Latin, digits, punctuation, repeated whitespace, Unicode variants, `ё`, null, empty.
- Determinism and idempotence of normalization.
- Raw column unchanged.
- Field-aware dense composition order and description truncation priority.
- Lemmatizer fixture with inflected Russian forms; no online download in tests.
- Cache invalidates when normalization/lemmatizer config changes.
- Same row through train and benchmark entrypoints yields same representations.

## Verification
```bash
uv run pytest tests/test_preprocessing.py -q
uv run python -m avito_candidate_generation.preprocessing materialize --config configs/smoke.toml
uv run python -m avito_candidate_generation.preprocessing materialize --config configs/base.toml
make ci
```
Assertions:
- no ID/order corruption;
- normalized output has no repeated whitespace;
- raw text retained;
- all truncation/missing counts reported;
- identical rerun yields same artifact fingerprint and cache hit;
- lemma artifacts exist only when enabled and identify backend/version.

## Expected Outputs / Artifacts
`artifacts/preprocessing/<run_id>/queries_*.parquet`, `items_*.parquet`, `text_stats.json`, config snapshot and manifest. Raw/normalized always; lemma only opt-in.

## Documentation Requirements
Explain normalization choices, model-specific prefixes, truncation priority, and why lemma remains a measured BM25 variant. Comments explain Russian-specific behavior, not loop syntax.

## Definition of Done
- Unit tests and `make ci` pass.
- Smoke/full materialization deterministic and cache-aware.
- No raw text loss; preprocessing config/version in every consumer-visible artifact.
- Missing/long input policies are explicit and measured.

## Do Not
- Не replace raw text in canonical data.
- Не fit preprocessing on validation labels.
- Не silently download models in unit tests.
- Не concatenate all fields irreversibly.
- Не claim lemma helps before stage `06` ablation.

## Handoff to Next Stage
Splits/evaluator and retrievers получают versioned, field-aware deterministic text representations and measured resource statistics.
