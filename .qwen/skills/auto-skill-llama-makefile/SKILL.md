---
name: llama-makefile
description: Create a Makefile for managing llama models with list, serve, interactive selection, venv management, and config file support
source: auto-skill
extracted_at: '2026-06-23T12:02:51.278Z'
---

## Creating a Makefile for llama Model Management

When the user wants to manage and serve llama models, create a Makefile with these targets. **Use the native `llama-server` binary** (from brew on macOS or pip on Linux) rather than the Python wrapper, as it handles quantized models more efficiently.

### Key Structure

```makefile
# Find llama-server binary (from pip or conda)
LLAMA_SERVER := $(shell command -v llama-server 2>/dev/null)
MODELS_DIR := $(HOME)/models
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Load configuration
-include config.mk
-include config.local.mk

.DEFAULT_GOAL := help

.PHONY: list serve serve-from-list help install venv venv-list venv-PYTHON

# List available Python versions
venv-list:
	@echo "Available Python versions:"
	@for py in python3 python3.13 python3.12 python3.11 python3.10; do \
		if command -v $$py >/dev/null 2>&1; then \
			$$py --version 2>&1; \
		fi; \
	done

# Create venv with default python3
venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)

# Create venv with specific Python version
venv-PYTHON:
ifndef PYTHON_VER
	$(error PYTHON_VER is not set. Run 'make venv-list' to see available versions)
endif
	@echo "Creating virtual environment with $(PYTHON_VER)..."
	$(PYTHON_VER) -m venv $(VENV_DIR)

# Install llama.cpp and Python dependencies (auto-creates venv)
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

# List available models
list:
	@echo "Available models in $(MODELS_DIR):"
	@ls -1 $(MODELS_DIR) 2>/dev/null || echo "  No models found"

# Serve a specific model (saves non-default PORT to config.local.mk)
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
	@if [ "$(PORT)" != "8000" ]; then \
		echo "PORT ?= $(PORT)" > config.local.mk; \
		echo "Saved port $(PORT) to config.local.mk"; \
	fi
	@echo "Starting llama server with model: $(MODEL)"
	@echo "  GPU layers: $(GPU_LAYERS), Context: $(CTX_SIZE), Max tokens: $(MAX_TOKENS), Port: $(PORT)"
	$(LLAMA_SERVER) --model $(MODELS_DIR)/$(MODEL) -ngl $(GPU_LAYERS) -c $(CTX_SIZE) -n $(MAX_TOKENS) --port $(PORT)

# Interactive model selection with port prompt (saves user-selected port)
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
	read -p "Enter port (default: $(PORT)): " port; \
	if [ -z "$$port" ]; then port=$(PORT); fi; \
	if [ "$$port" != "$(PORT)" ]; then \
		echo "PORT ?= $$port" > config.local.mk; \
		echo "Saved port $$port to config.local.mk"; \
	fi; \
	echo "Starting llama server with model: $$model on port $$port"; \
	$(LLAMA_SERVER) --model $(MODELS_DIR)/$$model -ngl $(GPU_LAYERS) -c $(CTX_SIZE) -n $(MAX_TOKENS) --port $$port

help:
	@echo "Available targets:"
	@echo "  venv-list        List available Python versions"
	@echo "  venv             Create Python virtual environment"
	@echo "  venv-PYTHON PYTHON_VER=X  Create venv with specific Python version"
	@echo "  install          Install llama.cpp (brew/pip) and Python dependencies"
	@echo "  list             List available models"
	@echo "  serve MODEL=<name>  Serve a specific model"
	@echo "  serve-from-list  Interactive model selection with port prompt"
	@echo "  help             Show this help message"
```

### Configuration Files

**config.mk** (committed to git, contains defaults):
```makefile
# Configuration for llama model serving
# Override these values in config.local.mk (gitignored)

# Model parameters
GPU_LAYERS ?= 70
CTX_SIZE ?= 131072
MAX_TOKENS ?= 8192

# Server parameters
PORT ?= 8000
```

**config.local.mk** (gitignored, contains user overrides):
```makefile
PORT ?= 9000
GPU_LAYERS ?= 90
```

The Makefile loads both files with `-include`, so `config.local.mk` overrides `config.mk`. When a user specifies a non-default port via `make serve PORT=9000` or enters it interactively, it's automatically saved to `config.local.mk` for future runs.

**.gitignore**:
```
.venv/
config.local.mk
```

### Important Details

1. **Use native llama-server binary** - The native binary (`llama-server`) handles quantized models (e.g., IQ2_M) much more efficiently than the Python wrapper (`python -m llama_cpp.server`). The Python wrapper can cause massive memory allocation errors (121+ GiB) with quantized models.

2. **Platform-specific installation** (follow official llama.cpp docs):
   - **macOS**: `brew install llama.cpp` (installs Homebrew first if needed)
   - **Linux**: `pip install llama-cpp-python`

3. **Find the binary dynamically** - Use `$(shell command -v llama-server 2>/dev/null)` to locate the binary from PATH.

4. **Check for binary before serving** - Add a check in serve targets to error gracefully if `llama-server` is not installed.

5. **Configuration via config.mk**:
   - Use `?=` for overridable defaults
   - Load with `-include config.mk` and `-include config.local.mk`
   - Auto-save user preferences (like port) to `config.local.mk`
   - Display current config values when starting server

6. **Common serve parameters** (configurable):
   - `-ngl $(GPU_LAYERS)` - GPU layers (default: 70)
   - `-c $(CTX_SIZE)` - Context size (default: 131072)
   - `-n $(MAX_TOKENS)` - Max tokens to predict (default: 8192)
   - `--port $(PORT)` - HTTP port (default: 8000)

7. **Interactive menu in Makefile** - Use shell scripting with proper escaping:
   - Double `$$` for shell variables in Make recipes
   - Use `read -p` for prompts
   - Use `sed -n "${choice}p"` to select the nth line

8. **Virtual environment management**:
   - `venv-list` - Shows available Python versions
   - `venv-PYTHON PYTHON_VER=python3.11` - Create venv with specific version
   - `install` - Auto-creates venv if missing, then installs from `pyproject.toml`

9. **pyproject.toml dependencies** - Include server dependencies:
   ```toml
   [project]
   name = "run-models"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = [
       "llama-cpp-python>=0.3.0",
       "uvicorn>=0.23.0",
       "fastapi>=0.100.0",
       "sse-starlette>=1.6.0",
       "starlette-context>=0.3.0",
       "anyio>=4.0.0",
       "pydantic-settings>=2.0.0",
   ]
   ```

10. **Cross-platform considerations** - User develops across Mac and Linux. Use `uname -s` for OS detection, `command -v` for path detection, avoid hardcoded paths.

11. **Default target** - Set `.DEFAULT_GOAL := help` so running `make` alone shows available commands.

12. **Error handling** - Use `ifndef MODEL` with `$(error ...)` to provide clear usage messages.

13. **Persist user preferences** - When user provides non-default values (e.g., port), save to `config.local.mk` so they persist across sessions.
