PYTHON := .venv/bin/python
export PYTHONPATH := src

.PHONY: test typecheck lint ci
test:
	$(PYTHON) -m pytest -q
lint:
	ruff check src tests
typecheck:
	$(PYTHON) -m pyright
ci: lint test typecheck
