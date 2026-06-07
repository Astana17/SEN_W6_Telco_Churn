PIP=python3 -m pip
PY=python3

.PHONY: help install train predict notebook setup clean

help:
	@echo "Available targets:"
	@echo "  make install   Install dependencies"
	@echo "  make train      Run scripts/train.py"
	@echo "  make predict    Run scripts/predict.py"
	@echo "  make notebook   Open notebook/EDA.ipynb"
	@echo "  make setup      install + train + predict"
	@echo "  make clean      Remove generated results"

install:
	@$(PIP) install -r requirements.txt

train:
	@$(PY) scripts/train.py

predict:
	@$(PY) scripts/predict.py

notebook:
	@$(PY) -m notebook notebook/EDA.ipynb

setup: install train predict

clean:
	@rm -rf results/plots results/*.csv results/*.json results/*.md results/*.pkl results/*.png
