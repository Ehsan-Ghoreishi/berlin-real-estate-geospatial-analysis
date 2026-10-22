.PHONY: help install install-dev lint format test run tune notebook clean

PYTHON ?= python3

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install dev dependencies and git hooks
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:  ## Run ruff lint and format checks
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint issues and format code
	ruff check --fix .
	ruff format .

test:  ## Run the test suite
	$(PYTHON) -m pytest

run:  ## Run the full pipeline
	$(PYTHON) main.py

tune:  ## Hyperparameter search for the gradient-boosting model (~3 min)
	$(PYTHON) -m src.tune

notebook:  ## Re-execute the analysis notebook in place (needs `make run` first)
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace berlin_market_analysis.ipynb

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
