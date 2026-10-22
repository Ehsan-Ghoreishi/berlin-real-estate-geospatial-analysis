.PHONY: help install install-dev lint typecheck format test run tune notebook docker-build docker-run site clean

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

typecheck:  ## Static type check with mypy
	mypy

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

docker-build:  ## Build the pipeline image
	docker build -t berlin-real-estate .

docker-run:  ## Run the full pipeline in Docker, writing outputs to the host
	docker run --rm -v "$(CURDIR)/data/processed:/app/data/processed" -v "$(CURDIR)/maps:/app/maps" berlin-real-estate

site:  ## Preview the GitHub Pages landing page at http://localhost:8000
	rm -rf _site && mkdir -p _site && cp -r site/. _site/ && cp -r maps _site/maps
	cd _site && $(PYTHON) -m http.server 8000

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache _site
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
