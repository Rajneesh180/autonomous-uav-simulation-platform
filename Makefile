.PHONY: help install test lint run batch clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies (pytest, ruff)
	pip install -r requirements.txt pytest ruff

test: ## Run the full test suite
	python -m pytest tests/ -v --tb=short

lint: ## Lint with ruff
	ruff check .

format: ## Auto-format with ruff
	ruff format .

run: ## Run a single simulation (headless)
	MPLBACKEND=Agg python main.py --mode single

run-gui: ## Run a single simulation with GUI
	python main.py --mode single --render

batch: ## Run a 10-seed batch experiment
	MPLBACKEND=Agg python main.py --mode batch

clean: ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
