# ============================================================
# DeFi Yield Aggregator – Makefile
# ============================================================
.PHONY: install dev test test-cov lint format typecheck check clean build \
       docker-build docker-test docker-run docker-shell \
       pre-commit-install pre-commit-run bump-version help

# ---------- Development ----------

install: ## Install the package in the current environment
	pip install -e .

dev: ## Install with dev/test dependencies
	pip install -e ".[dev]"

# ---------- Testing ----------

test: ## Run the test suite
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src/defi_yield_aggregator --cov-report=term-missing

# ---------- Code Quality ----------

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Auto-format code with ruff
	ruff format src/ tests/

typecheck: ## Run mypy type checker
	mypy src/defi_yield_aggregator/

check: lint typecheck test ## Run all checks (lint + typecheck + test)

# ---------- Pre-commit ----------

pre-commit-install: ## Install pre-commit hooks into .git/hooks
	pre-commit install

pre-commit-run: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

# ---------- Build & Release ----------

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean ## Build source and wheel distributions
	python -m build

bump-version: ## Bump patch version in pyproject.toml (e.g. 0.1.0 -> 0.1.1)
	@CURRENT=$$(grep -m1 'version = ' pyproject.toml | sed 's/.*"\(.*\)".*/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	PATCH=$$(echo $$CURRENT | cut -d. -f3); \
	NEW_PATCH=$$((PATCH + 1)); \
	NEW="$$MAJOR.$$MINOR.$$NEW_PATCH"; \
	echo "Bumping version: $$CURRENT -> $$NEW"; \
	sed -i "s/version = \"$$CURRENT\"/version = \"$$NEW\"/" pyproject.toml; \
	sed -i "s/__version__ = \"$$CURRENT\"/__version__ = \"$$NEW\"/" src/defi_yield_aggregator/__init__.py

# ---------- Docker ----------

docker-build: ## Build the Docker image
	docker compose build defi-yield

docker-test: ## Run the test suite inside a Docker container
	docker compose run --rm test

docker-run: ## Run the CLI inside Docker (pass args via ARGS)
	docker compose run --rm defi-yield $(ARGS)

docker-shell: ## Open an interactive shell inside the Docker container
	docker compose run --rm --entrypoint /bin/bash defi-yield

# ---------- Help ----------

help: ## Show this help message
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Docker usage examples:"
	@echo "  make docker-build"
	@echo "  make docker-run ARGS='pools --chain ethereum'"
	@echo "  make docker-run ARGS='optimize 10000 --max-risk 50'"
	@echo "  make docker-test"
