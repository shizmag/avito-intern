PYTHON := .venv/bin/python
export PYTHONPATH := src

.PHONY: test test-ml coverage format-check lint typecheck ci
CHANGED_PYTHON := $(shell git diff --name-only --diff-filter=ACM | grep -E '\\.py$$' || true)

test:
	$(PYTHON) -m pytest -q

test-ml:
	$(PYTHON) -m pytest tests/test_dense.py tests/test_two_tower.py tests/test_two_tower_hard.py tests/test_neural_retrieval.py tests/test_fusion.py tests/test_oof_catboost.py -q

coverage:
	$(PYTHON) -m pytest --cov=src/avito_candidate_generation --cov-report=term-missing --cov-fail-under=65 -q

format-check:
	@if test -n "$(CHANGED_PYTHON)"; then ruff format --check $(CHANGED_PYTHON); else echo "No changed Python files"; fi

lint:
	ruff check src tests

typecheck:
	$(PYTHON) -m pyright

ci: format-check lint typecheck test coverage
