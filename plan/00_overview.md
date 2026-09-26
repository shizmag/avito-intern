# Карта реализации Avito Candidate Generation

## Role
Вы — Lead ML Architect и координатор последовательной реализации. Используйте этот файл как обязательную карту работ; не реализуйте весь pipeline одним изменением.

## Goal
Организовать реализацию воспроизводимого candidate-generation pipeline от raw Parquet до проверенного `answer.csv`, оптимизирующего coverage релевантных items при бюджете 50. Каждый следующий файл — самостоятельный task prompt и отдельная проверяемая стадия.

## Context
Текущий репозиторий минимален: `eda.py` содержит marimo-EDA, `task.md` — исходное задание, `README.md` пуст, `main.py` — заглушка, `pyproject.toml` содержит только EDA/data-science зависимости, тестов и production pipeline нет. Данные находятся в `data/`: 497673 train rows, 2452 benchmark queries, 189212 benchmark items. Нельзя считать наблюдения EDA контрактами без автоматической проверки.

Целевая метрика — mean query Recall@50; порядок внутри top-50 не влияет. Архитектурная гипотеза:

```text
query + inference-safe fields
           |
 category strategy (global / hard / fallback, chosen by validation)
           |
   +-------+----------------+------------------+
   |                        |                  |
BM25 raw/lemma      generic dense       supervised two-tower
   |                        |                  |
   +---------------- candidate union ----------+
                            |
                   source ranks/scores + RRF
                            |
                  CatBoost OOF fusion selector
                            |
                    <=50 unique item_ids
                            |
                 submission validator -> answer.csv
```

Это гипотеза, не догма: каждый компонент допускается в final config только после leakage-safe ablation.

## Prerequisites
- Репозиторий доступен из корня проекта.
- `data/*.parquet`, `task.md`, `eda.py`, `pyproject.toml` доступны.
- Перед каждой стадией агент читает `AGENTS.md`, этот файл, prompt стадии, актуальные config/contracts и результаты предыдущих стадий.

## Inputs
- `data/train.parquet`
- `data/benchmark_queries.parquet`
- `data/benchmark_items.parquet`
- `task.md`, `eda.py`, `pyproject.toml`, `pyrightconfig.json`
- Outputs предыдущих стадий по dependency graph.

## Tasks
1. Выполнять файлы строго в основном порядке:
   1. `01_repository_foundation.md`
   2. `02_data_contracts_and_eda.md`
   3. `03_text_preprocessing.md`
   4. `04_cold_item_splits.md`
   5. `05_canonical_evaluation.md`
   6. `06_bm25_lexical_retrieval.md`
   7. `07_generic_dense_retrieval.md`
   8. `08_two_tower_baseline.md`
   9. `09_hard_negative_mining.md`
   10. `10_two_tower_hard_negative_training.md`
   11. `11_candidate_union_rrf_features.md`
   12. `12_oof_candidate_generation.md`
   13. `13_catboost_fusion.md`
   14. `14_ablation_and_model_selection.md`
   15. `15_final_training.md`
   16. `16_benchmark_inference.md`
   17. `17_submission_validation.md`
   18. `18_readme_and_final_report.md`
   19. `19_end_to_end_verification.md`
2. Останавливать downstream ML-работы при падении соответствующего quality gate; исключение — только независимая ветка, а блокер записан в manifest/report.
3. На каждой стадии сохранять config snapshot, seed, data/artifact fingerprints, dependency/model versions, git commit (если доступен), команды и metrics в предсказуемом run directory.
4. Не использовать benchmark labels или submission score как основной tuning loop. Максимум 7 внешних attempts — только final feedback после локального выбора.

Dependency graph:

```text
01 foundation
      |
02 data contracts + reproducible EDA
      +----------------------+
      |                      |
03 preprocessing        04 cold-item splits
      |                      |
      +----------+-----------+
                 |
          05 evaluator
          /          \
06 BM25 lexical     07 generic dense
          \          /
           08 two-tower baseline
                    |
           09 hard-negative mining
                    |
           10 hard-negative two-tower
                    |
        11 union + RRF + pair features
                    |
          12 OOF candidate/features
                    |
           13 CatBoost fusion
                    |
       14 ablation/model selection
                    |
             15 final fit
                    |
         16 benchmark inference
                    |
       17 submission validation
                    |
          18 README/report
                    |
        19 full reproducibility gate
```

Параллельность:
- После Gate 1: `06` и `07` можно выполнять параллельно.
- Подготовку training loop в `08` можно начать после `03`–`05`, но его baseline comparison ждёт outputs `07`.
- Документационные заметки для `18` ведутся на всех стадиях, но итоговый README блокируется `14`–`17`.
- Остальные стадии blocking и выполняются последовательно.

## Architecture / Design Constraints
- Canonical package: `src/avito_candidate_generation/`; не создавать второй конкурирующий pipeline.
- Config format: TOML в `configs/` (stdlib `tomllib` для чтения); параметры не размазывать константами.
- Canonical artifact layout: `artifacts/<stage>/<run_id>/`; большие артефакты не коммитить.
- IDs `query_id`, `item_id`, `internal_query_id` — строки на всех границах; никогда не приводить к integer/float.
- Query unit train: стабильный `internal_query_id` из полного query-feature tuple; query slices `seen/unseen` определяются отдельно по тексту.
- Full-corpus retrieval, не sampled-negative evaluation.
- Primary split — disjoint cold item; test используется один раз после выбора архитектуры.
- Core features берутся только из inference snapshot. Label/history aggregates — отдельная opt-in ветка с temporal-leakage warning, не final default.
- Benchmark dense retrieval exact и батчированный. ANN запрещён без отдельной production-only rationale.
- Candidate K должен быть >50; selection objective — Recall@50.
- OOF retrieval features обязательны для CatBoost. Не inject пропущенные positives в candidate tables.
- Category hard filter и lemmatization принимаются только по validation; location hard filter по умолчанию запрещён.
- Heavy operations имеют `smoke` mode, batching, cache/fingerprint и resume-safe output.

Контракты между стадиями:

| Producer | Contract artifact | Consumers |
|---|---|---|
| 01 | config/run manifest/CLI conventions, CI | все |
| 02 | validated canonical tables, schema + EDA report | 03–19 |
| 03 | deterministic raw/normalized/lemma representations | retrievers/features |
| 04 | split assignments, query ground truth, leakage report | 05–15 |
| 05 | canonical evaluator and metrics schema | все эксперименты |
| 06/07/08/10 | unified candidate Parquet | 09, 11–14 |
| 09 | reproducible negative Parquet | 10, 15 |
| 11 | union/pair-feature schema and RRF | 12–16 |
| 12 | OOF candidate feature table | 13 |
| 13 | fusion model + evaluated predictions | 14–16 |
| 14 | immutable selected config and decision report | 15–19 |
| 15 | final model/index artifacts + manifest | 16 |
| 16 | raw benchmark predictions | 17 |
| 17 | validated `answer.csv` + validation report | 18–19 |

Quality gates:
- **Gate 1 — correctness:** schemas, ID preservation, deterministic preprocessing, item-disjoint splits, metric tests, `make ci` pass.
- **Gate 2 — lexical baseline:** BM25 raw R@50 and candidate-pool R@200/R@500 known; category strategy and lemma impact measured.
- **Gate 3 — dense retrieval:** generic dense and two-tower measured on same corpus/split; union oracle coverage known; hard-negative effect measured.
- **Gate 4 — fusion:** RRF baseline and leakage-safe CatBoost OOF evaluation known; validation gain and slice regressions reported.
- **Gate 5 — submission:** selected pipeline refit, exact benchmark inference complete, `answer.csv` round-trip validator pass, README and clean-clone smoke reproduction pass.

## Files to Create or Modify
- `plan/00_overview.md` through `plan/19_end_to_end_verification.md` are specifications.
- Production files are created only by agents executing stages, under `src/`, `configs/`, `tests/`, `scripts/` only where CLI wrapper is justified, `artifacts/`, `reports/`, `README.md`, `Makefile`, `pyproject.toml`.

## Interfaces / Data Contracts
Shared candidate schema required from all retrievers:

```text
internal_query_id: string (or benchmark query_id mapped to this field)
item_id: string
source: string
score: float64, finite
rank: int32, 1-based contiguous per query/source
```

Shared metrics JSON minimum:

```json
{"stage":"...","run_id":"...","split":"validation","seed":42,"metrics":{"recall@50":0.0},"slices":{},"counts":{},"timings":{},"status":"PASS|FAIL"}
```

Every Parquet artifact gets schema version and fingerprint in adjacent manifest.

## Tests
This overview introduces no production tests. Every executable stage must add focused unit/integration tests and keep prior tests green. Heavy model downloads/GPU are excluded from default unit tests through fakes and explicit markers.

## Verification
Run after plan creation:

```bash
find plan -maxdepth 1 -name '*.md' | sort
for f in plan/*.md; do
  for h in '## Role' '## Goal' '## Context' '## Prerequisites' '## Inputs' '## Tasks' '## Architecture / Design Constraints' '## Files to Create or Modify' '## Interfaces / Data Contracts' '## Tests' '## Verification' '## Expected Outputs / Artifacts' '## Documentation Requirements' '## Definition of Done' '## Do Not' '## Handoff to Next Stage'; do
    grep -Fq "$h" "$f" || echo "MISSING $h in $f"
  done
done
```

Expected: 20 files, no `MISSING` lines, numbering `00`–`19` contiguous.

## Expected Outputs / Artifacts
- Однозначная последовательность исполнения.
- Явный dependency graph, quality gates и cross-stage schemas.
- Machine-readable stage outputs во время будущей реализации.

## Documentation Requirements
Каждая стадия объясняет reasoning: почему full-corpus evaluation, cold-item split, exact benchmark search, category ablation, non-hard location, candidate K>50, hard negatives и OOF. Комментарии не должны пересказывать синтаксис.

## Definition of Done
- Все 20 prompts существуют и проходят heading check.
- Каждый downstream contract имеет ровно одного канонического producer.
- Все пути от raw parquet до validated `answer.csv` покрыты.
- Blocking gates и допустимая параллельность явно указаны.

## Do Not
- Не исполнять все prompts одним агентом/коммитом.
- Не менять contracts молча; изменение schema требует version bump и обновления consumers/tests.
- Не считать существующий EDA доказательством без executable report.
- Не продвигаться через failed gate с формулировкой «скорее всего работает».

## Handoff to Next Stage
Агент `01_repository_foundation.md` может считать структуру стадий, config/artifact conventions и общие запреты зафиксированными. Он должен создать минимальный reproducibility/CI skeleton, но не реализовывать retrieval.
