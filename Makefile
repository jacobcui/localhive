# localhive - run local GGUF models sized to the machine you are on.
#
#   make list-models              show only models that fit THIS machine
#   make install-model MODEL=X    download model X into ~/.localhive/models
#   make serve MODEL=X            run llama-server for an installed model
#   make install                  install llama.cpp + python deps
#
# Everything is driven from the repo root so it works the same on
# WSL2 Ubuntu, native Linux and macOS.

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

MODELS_DIR  ?= $(HOME)/.localhive/models
VENV        ?= .venv
PYTHON      ?= $(VENV)/bin/python
LOCALHIVE   := $(VENV)/bin/localhive

# Context to plan for when deciding what fits. Override like:
#   make list-models CONTEXT=32768
CONTEXT     ?= 8192

.DEFAULT_GOAL := help
.PHONY: help install deps llama check list-models list install-model installed \
        serve hardware where clean-venv doctor test

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
help: ## Show this help
	@echo "localhive - machine-aware local model manager"
	@echo ""
	@echo "Usage: make <target> [MODEL=<name>] [CONTEXT=<tokens>]"
	@echo ""
	@echo "  MODELS_DIR = $(MODELS_DIR)"
	@echo "  CONTEXT    = $(CONTEXT)"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical flow:"
	@echo "  make install              # once: llama.cpp + venv"
	@echo "  make list-models          # see what this box can run"
	@echo "  make install-model MODEL=qwen3.6-35b-a3b-iq2m"
	@echo "  make serve MODEL=qwen3.6-35b-a3b-iq2m"

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
deps: ## Install Python virtualenv + localhive into it
	@if [ ! -d "$(VENV)" ]; then \
		echo "==> creating virtualenv at $(VENV)"; \
		python3 -m venv "$(VENV)"; \
	fi
	@echo "==> installing python deps"
	@"$(VENV)/bin/pip" install --quiet --upgrade pip
	@"$(VENV)/bin/pip" install --quiet -e .
	@echo "==> done"

llama: ## Install llama.cpp (brew on macOS, prebuilt binary on Linux/WSL)
	@bash scripts/install_llama.sh

install: llama deps ## Install llama.cpp and all Python dependencies

check: ## Verify llama-server and localhive are usable
	@bash scripts/check_env.sh

doctor: check ## Alias for check

test: ## Run the test suite
	@$(PYTHON) -m pytest -q tests 2>/dev/null || $(PYTHON) tests/test_fit.py

# ---------------------------------------------------------------------------
# the two important targets
# ---------------------------------------------------------------------------
list-models: deps ## List models that fit this machine
	@$(LOCALHIVE) list-models --context $(CONTEXT) $(ARGS)

list: list-models ## Alias for list-models

install-model: deps ## Install a model (MODEL=<name from list-models>)
ifndef MODEL
	$(error MODEL is not set. Run 'make list-models', then 'make install-model MODEL=<name>')
endif
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" install-model "$(MODEL)" --context $(CONTEXT) $(ARGS)

installed: deps ## Show models already installed
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" list-installed

serve: deps ## Serve an installed model (MODEL=<name>)
ifndef MODEL
	$(error MODEL is not set. Run 'make installed', then 'make serve MODEL=<name>')
endif
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" serve "$(MODEL)" $(ARGS)

hardware: deps ## Show detected CPU / RAM / GPU
	@$(LOCALHIVE) hardware $(ARGS)

where: ## Print the models directory
	@echo "$(MODELS_DIR)"

# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
clean-venv: ## Remove the virtualenv
	@rm -rf "$(VENV)"
	@echo "removed $(VENV)"
