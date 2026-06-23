# Makefile for managing llama models
# Helper for running llama app (@/home/jacob/miniforge3/bin/llama)
# Models are stored in ~/models directory

LLAMA := /home/jacob/miniforge3/bin/llama
MODELS_DIR := $(HOME)/models
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.DEFAULT_GOAL := help

.PHONY: list serve serve-from-list help install venv

venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Run 'make install' to install llama-cpp-python"

install: $(VENV_DIR)/bin/activate
	@echo "Installing llama-cpp-python..."
	$(PIP) install llama-cpp-python

$(VENV_DIR)/bin/activate:
	@echo "Virtual environment not found. Creating..."
	@$(MAKE) venv

help:
	@echo "Available targets:"
	@echo "  venv             Create Python virtual environment"
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
	$(PYTHON) -m llama_cpp.server --model $(MODELS_DIR)/$(MODEL) --n-gpu-layers 999 --ctx-size 131072 --n-predict 8192

serve-from-list:
	@echo "Available models:"
	@i=1; for m in $$(ls -1 $(MODELS_DIR) 2>/dev/null); do echo "  $$i) $$m"; i=$$((i+1)); done
	@echo ""
	@read -p "Select model (1-$$((i-1))): " choice; \
	if [ -z "$$choice" ]; then echo "No selection made"; exit 1; fi; \
	model=$$(ls -1 $(MODELS_DIR) 2>/dev/null | sed -n "$${choice}p"); \
	if [ -z "$$model" ]; then echo "Invalid selection: $$choice"; exit 1; fi; \
	echo "Starting llama server with model: $$model"; \
	$(PYTHON) -m llama_cpp.server --model $(MODELS_DIR)/$$model --n-gpu-layers 999 --ctx-size 131072 --n-predict 8192
