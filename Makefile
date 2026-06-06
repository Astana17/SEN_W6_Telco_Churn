PIP=python3 -m pip
PY=python3

.PHONY: help install train predict notebook setup script-setup clean

help:
	@echo "Available targets:"
	@echo "  make install        Install dependencies"
	@echo "  make train          Run scripts/train.py"
	@echo "  make predict        Run scripts/predict.py"
	@echo "  make notebook       Open notebook/EDA.ipynb"
	@echo "  make setup          install + train + predict"
	@echo "  make script-setup   Run scripts/setup_and_run.sh"
	@echo "  make clean          Remove generated results"

install:
	@echo "[make] Upgrading pip..."
	@$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then \
		echo "[make] Installing from requirements.txt..."; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "[make] requirements.txt not found, skipping"; \
	fi
	@echo "[make] Ensuring notebook is installed..."
	@$(PIP) install notebook

train:
	@echo "[make] Running training..."
	@$(PY) scripts/train.py

predict:
	@echo "[make] Running prediction..."
	@$(PY) scripts/predict.py

notebook:
	@echo "[make] Launching notebook..."
	@$(PY) -m notebook notebook/EDA.ipynb

setup: install train predict

script-setup:
	@bash scripts/setup_and_run.sh

clean:
	@echo "[make] Cleaning results..."
	@rm -f results/churn_pipeline.pkl results/predictions.csv results/results.md
	@rm -f results/plots/*.png