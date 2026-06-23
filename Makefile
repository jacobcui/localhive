# Makefile for managing llama models
# Uses llama-server binary from llama.cpp
# Models are stored in ~/models directory
#
# Configuration:
#   - server.conf: llama-server arguments (edit this file)
#
# Installation:
#   - macOS: brew install llama.cpp
#   - Linux: pip install llama-cpp-python

# Find llama-server binary (from pip or conda)
LLAMA_SERVER := $(shell command -v llama-server 2>/dev/null)
MODELS_DIR := $(HOME)/models
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
CONFIG_FILE := server.conf

# Load configuration file content (strip comments and empty lines)
SERVER_ARGS := $(shell cat $(CONFIG_FILE) 2>/dev/null | grep -v '^#' | grep '.')

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

install:
	@echo "Installing llama.cpp and Python dependencies..."
	@OS=$$(uname -s); \
	if [ "$$OS" = "Darwin" ]; then \
		echo "macOS detected. Installing llama.cpp via Homebrew..."; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "Homebrew not found. Installing Homebrew..."; \
			/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		fi; \
		brew install llama.cpp; \
	elif [ "$$OS" = "Linux" ]; then \
		echo "Linux detected. Installing llama-cpp-python via pip..."; \
		pip install llama-cpp-python; \
	else \
		echo "Unsupported OS: $$OS"; \
		exit 1; \
	fi
	@echo ""
	@echo "Setting up Python virtual environment..."
	@if [ ! -d $(VENV_DIR) ]; then $(MAKE) venv; fi
	@echo ""
	@echo "Installing Python dependencies..."
	$(PIP) install -e .
	@echo ""
	@echo "Verifying installation..."
	@llama-server --version || echo "Warning: llama-server not found in PATH"

$(VENV_DIR)/bin/activate:
	@echo "Virtual environment not found. Creating..."
	@$(MAKE) venv

help:
	@echo "Available targets:"
	@echo "  venv-list        List available Python versions"
	@echo "  venv             Create Python virtual environment (default python3)"
	@echo "  venv-PYTHON PYTHON_VER=X Create venv with specific Python version"
	@echo "  install          Install llama.cpp (brew/pip) and Python dependencies"
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
	@if ! command -v llama-server >/dev/null 2>&1; then \
		echo "Error: llama-server binary not found."; \
		echo "Install via:"; \
		echo "  - macOS:  brew install llama.cpp"; \
		echo "  - Linux:  pip install llama-cpp-python"; \
		exit 1; \
	fi
	@echo "Starting llama server with model: $(MODEL)"
	@echo "  Args: $(SERVER_ARGS)"
	$(LLAMA_SERVER) --model $(MODELS_DIR)/$(MODEL) $(SERVER_ARGS)

serve-from-list:
	@if ! command -v llama-server >/dev/null 2>&1; then \
		echo "Error: llama-server binary not found."; \
		echo "Install via:"; \
		echo "  - macOS:  brew install llama.cpp"; \
		echo "  - Linux:  pip install llama-cpp-python"; \
		exit 1; \
	fi
	@echo "Available models:"
	@i=1; for m in $$(ls -1 $(MODELS_DIR) 2>/dev/null); do echo "  $$i) $$m"; i=$$((i+1)); done
	@echo ""
	@read -p "Select model (1-$$((i-1))): " choice; \
	if [ -z "$$choice" ]; then echo "No selection made"; exit 1; fi; \
	model=$$(ls -1 $(MODELS_DIR) 2>/dev/null | sed -n "$${choice}p"); \
	if [ -z "$$model" ]; then echo "Invalid selection: $$choice"; exit 1; fi; \
	echo "Starting llama server with model: $$model"; \
	$(LLAMA_SERVER) --model $(MODELS_DIR)/$$model $(SERVER_ARGS)
