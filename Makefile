# AeroNet AI quick commands.

PYTHON ?= python

.PHONY: help install test train backup run clean

help: ## Show available targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install Python dependencies.
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test: ## Run the test suite (pytest).
	$(PYTHON) -m pytest -q

train: ## Train the classification heads and evaluate.
	$(PYTHON) scripts/train_models.py

backup: ## Create a dated local backup.
	$(PYTHON) backup_project.py

run: ## Run the API locally.
	$(PYTHON) app.py

clean: ## Remove Python cache files.
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
