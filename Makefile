# Makefile for managing llama models
# Helper for running llama app (@/home/jacob/miniforge3/bin/llama)
# Models are stored in ~/models directory

LLAMA := /home/jacob/miniforge3/bin/llama
MODELS_DIR := $(HOME)/models
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.DEFAULT_GOAL := help

.PHONY: list serve serve-from-list help install venv venv-list

venv-list:
	@echo "Available Python versions:"
	@for py in python3 python3.13 python3.12 python3.11 python3.10; do \
		if command -v $$py >/dev/null 2>&1; then \
			$$py --version 2>&1; \
		fi; \
	done

venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Run 'make install' to install dependencies from pyproject.toml"

venv-PYTHON:
ifndef PYTHON_VER
	$(error PYTHON_VER is not set. Run 'make venv-list' to see available versions, then 'make venv-PYTHON PYTHON_VER=python3.11')
endif
	@echo "Creating virtual environment with $(PYTHON_VER)..."
	$(PYTHON_VER) -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Run 'make install' to install dependencies from pyproject.toml"

install: $(VENV_DIR)/bin/activate
	@echo "Installing project dependencies..."
	$(PIP) install -e .

$(VENV_DIR)/bin/activate:
	@echo "Virtual environment not found. Creating..."
	@$(MAKE) venv

help:
	@echo "Available targets:"
	@echo "  venv-list        List available Python versions"
	@echo "  venv             Create Python virtual environment (default python3)"
	@echo "  venv-PYTHON PYTHON_VER=X Create venv with specific Python version"
	@echo "  install          Create venv (if needed) and install llama-cpp-python"
	@echo "  list             List available models in $(MODELS_DIR)"
	@echo "  serve MODEL=<name>  Start serving a specific model"
	@echo "  serve-from-list  Interactive menu to select and serve a model"
	@echo "  help             Show this help message"

list:
	@echo "Available models in $(MODELS_DIR):"
	@ls -1 $(MODELS_DIR) 2>/dev/null || echo "  No models found"

serve:
ifndef MODEL
	$(error MODEL is not set. Usage: make serve MODEL=<model_name>)
endif
	@echo "Starting llama server with model: $(MODEL)"
	$(PYTHON) -m llama_cpp.server --model $(MODELS_DIR)/$(MODEL) --n_gpu_layers 999 --n_ctx 131072

serve-from-list:
	@echo "Available models:"
	@i=1; for m in $$(ls -1 $(MODELS_DIR) 2>/dev/null); do echo "  $$i) $$m"; i=$$((i+1)); done
	@echo ""
	@read -p "Select model (1-$$((i-1))): " choice; \
	if [ -z "$$choice" ]; then echo "No selection made"; exit 1; fi; \
	model=$$(ls -1 $(MODELS_DIR) 2>/dev/null | sed -n "$${choice}p"); \
	if [ -z "$$model" ]; then echo "Invalid selection: $$choice"; exit 1; fi; \
	echo "Starting llama server with model: $$model"; \
	$(PYTHON) -m llama_cpp.server --model $(MODELS_DIR)/$$model --n_gpu_layers 999 --n_ctx 131072
