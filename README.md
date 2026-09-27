# Avito Candidate Generation

Гибридный candidate generator для задания Avito Search. Целевая метрика — **Recall@50**: для каждого запроса нужно вернуть не более 50 уникальных `item_id`, причём порядок внутри этих 50 не влияет на официальную метрику.

## TL;DR

Решение строит кандидатов из трёх независимых сигналов: lexical BM25, локальный multilingual dense encoder и task-specific trainable Two-Tower. Списки объединяются, затем применяется deterministic RRF, после чего сохраняются top-50 кандидатов. Основная validation — deterministic cold-item split: item IDs validation не пересекаются с item IDs, использованными для fit.

Для этого offline benchmark exact dense search — сознательный выбор: корпус около 189k объявлений, Recall@50 важнее approximation speed, а матричный exact search не добавляет ANN recall loss. В production-сценарии с миллионами items и высоким QPS тот же dense signal логично заменить на ANN index.

Текущий репозиторий содержит engineering-complete workflow и smoke/E2E проверки. Настоящий champion должен фиксироваться только после real-data ablation table; synthetic smoke metrics не являются доказательством качества.

## Архитектура

```text
query
  │
  ├── BM25
  ├── generic dense encoder
  └── Two-Tower query tower
          │
      candidate union
          │
       RRF fusion
          │
       deterministic top-50
          │
       answer.csv
```

Selected config: `configs/selected.toml`. Сейчас research/selection policy явно записывает `category_policy = "none"`; location не используется как unconditional hard filter.

## Задача и метрика

- input: search query + query filters, item snapshot;
- output: до 50 items на query;
- primary metric: `Recall@50`;
- diagnostics: `Recall@10/20/100/200/500`, candidate coverage, slice metrics.

Recall@50 считается как query-mean recall по ground-truth pairs cold-item validation. Не смешивать его с Recall@20, NDCG или MRR.

## Почему cold-item validation

Random row split переоценивает качество: одинаковые query texts и item IDs повторяются в train. Поэтому подготовленный split детерминированно распределяет item IDs по `train/validation/test`, а validation ground truth строится только по held-out validation items.

Проверка текущих подготовленных artifacts:

- `train_items ∩ validation_items = ∅`;
- validation ground truth: 43,413 queries и 46,360 relevant pairs;
- benchmark: 2,452 queries и 189,212 items — это inventory, не relevance score.

Полезные EDA facts:

- train: 497,673 interaction rows, 74,529 query texts, 344,825 unique train item IDs;
- benchmark query texts seen in train: около 37%;
- category agreement на train positive rows: около 99.99%;
- location agreement: около 83%.

Следствие: category restriction можно исследовать отдельно, но текущий selected config не объявляет hard filter победителем; location hard filter не используется.

## Retrievers и fusion

### BM25

Dependency-light lexical baseline. Реализация использует inverted postings, deterministic tie-break по `item_id` и fallback на zero-score items. Это базовый контроль качества и отдельный real-validation row.

### Dense

Локальная Hugging Face transformer model. Embeddings normalized; exact top-k выполняется батчами и сохраняется в artifacts. Runtime использует `local_files_only=True`: train/inference не скачивают модель молча.

### Two-Tower

Настоящий PyTorch query/item model с in-batch negatives. Workflow сохраняет v1 checkpoint, добывает hard negatives и обучает v2. Item encoding выполняется батчами, а сохранённые item embeddings переиспользуются retrieval stages.

### RRF

RRF объединяет complementary candidate lists без дополнительного learned selector. CatBoost/OOF primitives остаются research surface, но не считаются selected component без одинакового real validation protocol и доказуемого выигрыша Recall@50.

## Validation strategy и честный model selection

Минимальная real table должна содержать одинаковые validation corpus/query population и:

- BM25 raw;
- generic dense;
- Two-Tower v1;
- Two-Tower v2 hard-negative;
- unions на K=100/200/300/500;
- RRF;
- CatBoost только с корректными OOF features;
- category `none`, hard filter и fallback variants, если policy реализована.

Команда `real-validation` сейчас запускает безопасный BM25-only baseline и сохраняет:

```text
artifacts/metrics/real_validation.json
artifacts/metrics/real_validation.csv
```

Она не подменяет neural ablation table. Пока full neural matrix не выполнена, не утверждать, что текущая BM25+dense+Two-Tower+RRF комбинация — champion.

Observed real-data BM25 baseline (`cold-item-validation`, 43,413 queries, 34,370 validation items):

| Method | Recall@10 | Recall@20 | Recall@50 | Recall@200 | Recall@500 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1826 | 0.2627 | **0.4000** | 0.6541 | 0.7891 |

Это единственный сохранённый real ranking result на текущем run. Dense, Two-Tower, union/RRF, category variants и CatBoost не считаются измеренными: их artifacts отсутствуют, а release verifier блокирует selected pipeline.

## Point-in-time correctness

Query fields из benchmark query table и item snapshot считаются доступными во время inference. Core retrieval не использует сомнительные future aggregates. Любые historical features должны соблюдать `feature_time <= prediction_time`; иначе они не годятся для production evaluation.

## Exact search vs ANN

Benchmark solution оптимизирован под quality-first offline inference:

- корпус порядка 189k items;
- exact normalized matrix multiplication;
- deterministic top-k и отсутствие approximation loss;
- persisted embeddings.

Production Avito-like system при millions of items, high QPS и strict latency использовал бы offline item embeddings + ANN index + online query tower + lightweight fusion. ANN сознательно не добавлен перед release: это другая инженерная trade-off, не обязательная для данного offline задания.

## Структура проекта

```text
configs/                         TOML configs; selected.toml — default
src/avito_candidate_generation/
  retrievers/                    BM25, dense, exact search, Two-Tower
  training/                      contrastive and hard-negative training
  fusion/                        RRF and CatBoost/OOF primitives
  workflow.py                    selected artifact workflow
  provisioning.py                data/model preflight
  validation.py                  memory-bounded real validation
  submission.py                  answer.csv writer + validator
artifacts/                       ignored generated outputs
reports/                         methodology notes
plan/                            implementation contracts
```

## Быстрый старт

Требуется Python 3.13 и `uv`.

```bash
uv sync
uv run avito --help
```

Положить входные файлы в `data/`:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Проверить данные и подготовить модель:

```bash
uv run avito validate --config configs/selected.toml
uv run avito prepare --config configs/selected.toml
```

`prepare` работает offline по умолчанию и проверяет `configs/selected.toml`. Если model directory отсутствует, команда завершается fail-fast. Для разрешённого одноразового скачивания pinned open-source model:

```bash
uv run avito prepare --config configs/selected.toml --allow-network
```

После provisioning модель лежит в `artifacts/models/paraphrase-multilingual-MiniLM-L12-v2/`; там же появляется `model_manifest.json` с revision и SHA-256 файлов. Можно вручную положить заранее скачанную модель в этот путь и снова запустить `prepare` без сети.

## Полный запуск

```bash
uv run avito train
uv run avito evaluate
uv run avito predict --output answer.csv
uv run avito verify --submission answer.csv \
  --queries data/benchmark_queries.parquet \
  --items data/benchmark_items.parquet
```

`train` и neural retrieval — тяжёлые offline stages. GPU не обязателен для кода, но для полного dense/tower run рекомендуется accelerator. Повторный запуск использует prepared-data и embedding cache при совпадении fingerprints/config/model revision. Smoke path остаётся дешёвым wiring check:

```bash
uv run avito smoke
```

Smoke не является quality validation.

## Артефакты

Selected workflow пишет в `artifacts/selected/`:

- `prepared/manifest.json` — SHA-256 inputs, seed, outputs и cache contract;
- `prepared/*.parquet` — canonical train/validation/benchmark tables;
- `lexical/bm25.npz`;
- `embeddings/**` — normalized embedding artifacts;
- `two_tower/**` — safetensors checkpoints и hard negatives;
- `metrics/validation.json`;
- `predictions.parquet`;
- `manifest.json` — selected retrievers, RRF, Recall@50 metric name, config hash, git commit, model revision, category policy and split provenance.

Не коммитить data/models/embeddings/answer.csv.

## Генерация и проверка answer.csv

`predict` не редактирует CSV вручную: он читает persisted predictions, сохраняет `query_id,answer`, а writer выполняет round-trip validation. Independent check:

```bash
uv run avito verify --submission answer.csv \
  --queries data/benchmark_queries.parquet \
  --items data/benchmark_items.parquet
```

Проверяются UTF-8, ровно две колонки, query set/order, число строк, до 50 unique item IDs на query, существование item IDs, отсутствие duplicates и сохранение string IDs.

## Tests и code quality

```bash
make ci
make test-ml
uv run avito smoke
```

CI gate: Ruff format/lint, Pyright, pytest, coverage gate и deterministic fixture E2E. Full real training намеренно не входит в CI.

## Reproducibility

Фиксируются:

- TOML config hash;
- seed=42;
- cold-item split;
- git commit в selected manifest;
- input SHA-256 в `prepared/manifest.json`;
- model ID + immutable revision + local file hashes в `model_manifest.json`;
- normalized embedding metadata and fingerprints;
- deterministic tie-breaks (`score DESC, item_id ASC`).

`~/.cache/huggingface` не является silent runtime dependency: runtime принимает локальный configured path и падает, если model не provisioned.

## Open-source модели и библиотеки

- PyTorch — trainable Two-Tower;
- Transformers — local encoder loading;
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, purpose: multilingual dense text representation; model page: <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>;
- CatBoost — optional learned fusion primitive;
- NumPy/Pandas/PyArrow — arrays, tables, Parquet;
- safetensors — checkpoint serialization;
- local BM25 implementation — lexical baseline.

Внешний inference API не используется.

## Ограничения

- Полная neural real-data ablation matrix и окончательный champion пока требуют отдельного controlled run; smoke metrics нельзя выдавать за benchmark quality.
- CatBoost selected only after OOF leakage audit and stable Recall@50 improvement.
- Benchmark query labels не входят в supplied benchmark tables, поэтому benchmark prediction не равен benchmark quality evaluation.
- Exact dense search не подходит для production-scale millions/QPS; там нужен ANN.
