PYTHON := .venv/bin/python
export PYTHONPATH := src

.PHONY: test test-ml coverage format-check lint typecheck ci
test:
	$(PYTHON) -m pytest -q

test-ml:
	$(PYTHON) -m pytest tests/test_dense.py tests/test_two_tower.py tests/test_two_tower_hard.py tests/test_neural_retrieval.py tests/test_fusion.py tests/test_oof_catboost.py -q

coverage:
	$(PYTHON) -m pytest --cov=src/avito_candidate_generation --cov-report=term-missing --cov-fail-under=65 -q

format-check:
	ruff format --check main.py src/avito_candidate_generation/cli.py src/avito_candidate_generation/end_to_end.py src/avito_candidate_generation/fusion/catboost.py src/avito_candidate_generation/pipeline.py src/avito_candidate_generation/retrievers/dense.py src/avito_candidate_generation/retrievers/exact_search.py src/avito_candidate_generation/retrievers/two_tower.py src/avito_candidate_generation/training/two_tower.py src/avito_candidate_generation/verify.py tests/test_candidates_evaluation.py tests/test_end_to_end.py tests/test_neural_retrieval.py tests/test_oof_catboost.py tests/test_selected_pipeline.py tests/test_verify.py

lint:
	ruff check src tests

typecheck:
	$(PYTHON) -m pyright

ci: format-check lint typecheck test coverage
