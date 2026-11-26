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
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
LOCALHIVE   := $(VENV)/bin/localhive

# Stamp file: its timestamp records the last successful dependency install.
# It only goes stale when pyproject.toml changes or the venv is removed,
# so 'make list-models' does NOT re-run pip every time.
STAMP       := $(VENV)/.localhive-installed

# Context to plan for when deciding what fits. Override like:
#   make list-models CONTEXT=32768
CONTEXT     ?= 8192

# Content filter. Standard (safety-tuned / vendor) models are the default.
#   make list-models uncensored=yes     uncensored builds only
#   make list-models all=yes            both together
uncensored  ?= no
all         ?= no

ifeq ($(uncensored),yes)
CONTENT_FLAG := --uncensored
else ifeq ($(all),yes)
CONTENT_FLAG := --all-content
else ifeq ($(uncensored),no)
CONTENT_FLAG :=
else
$(error unknown uncensored=$(uncensored) - use uncensored=yes or all=yes)
endif

.DEFAULT_GOAL := help
.PHONY: help install deps llama link unlink check list-models list install-model \
        installed serve ps stop restart hardware where clean-venv doctor doctor-deps \
        test reinstall

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
	@echo "  content    = $(if $(CONTENT_FLAG),$(CONTENT_FLAG),standard models only (default))"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical flow:"
	@echo "  make install              # once: llama.cpp + venv + PATH"
	@echo "  make list-models          # see what this box can run"
	@echo "  make list-models uncensored=yes   # uncensored builds only"
	@echo "  make install-model MODEL=qwen3.6-35b-a3b-iq2m"
	@echo "  make serve MODEL=qwen3.6-35b-a3b-iq2m"
	@echo "  make ps                   # what is running?"
	@echo "  make stop all=yes         # stop everything"

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
# The venv and the install stamp are real files, so make can tell whether
# the work is already done. Nothing here runs on a plain 'make list-models'.
#
# scripts/bootstrap_venv.sh does the work and verifies each step, because
# 'python3 -m venv' can succeed on Debian/Ubuntu while producing a venv with
# no pip at all (when python3-venv is missing). Guarding on the python
# binary alone let that through and failed later with a confusing
# ".venv/bin/pip: No such file or directory".
$(VENV)/bin/python:
	@VENV="$(VENV)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/bootstrap_venv.sh

# Re-runs only when pyproject.toml (or the venv itself) is newer than the stamp.
$(STAMP): pyproject.toml $(VENV)/bin/python
	@VENV="$(VENV)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/bootstrap_venv.sh

deps: $(STAMP) ## Install Python virtualenv + localhive into it (idempotent)

reinstall: ## Force a fresh dependency install
	@rm -f "$(STAMP)"
	@$(MAKE) --no-print-directory deps

llama: ## Install llama.cpp (brew on macOS, prebuilt binary on Linux/WSL)
	@bash scripts/install_llama.sh

link: $(STAMP) ## Put 'localhive' on your PATH (via ~/.local/bin)
	@bash scripts/link_cli.sh || true

unlink: ## Remove the 'localhive' shim from ~/.local/bin
	@rm -f "${LOCALHIVE_BIN_DIR:-$(HOME)/.local/bin}/localhive"
	@echo "removed localhive shim"

install: llama deps link ## Install llama.cpp, deps, and put localhive on PATH

check: ## Verify llama-server and localhive are usable
	@bash scripts/check_env.sh

doctor: check ## Alias for check

test: $(STAMP) ## Run the test suite
	@$(PYTHON) -m pytest -q tests 2>/dev/null || $(PYTHON) tests/test_fit.py

doctor-deps: ## Diagnose a broken virtualenv
	@VENV="$(VENV)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/bootstrap_venv.sh

# ---------------------------------------------------------------------------
# the two important targets
# ---------------------------------------------------------------------------
list-models: $(STAMP) ## List models that fit this machine [uncensored=yes|all=yes]
	@$(LOCALHIVE) list-models --context $(CONTEXT) $(CONTENT_FLAG) $(ARGS)

list: list-models ## Alias for list-models

install-model: $(STAMP) ## Install a model (MODEL=<name from list-models>)
ifndef MODEL
	$(error MODEL is not set. Run 'make list-models', then 'make install-model MODEL=<name>')
endif
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" install-model "$(MODEL)" --context $(CONTEXT) $(CONTENT_FLAG) $(ARGS)

installed: $(STAMP) ## Show models already installed
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" list-installed

serve: $(STAMP) ## Serve an installed model (MODEL=<name>)
ifndef MODEL
	$(error MODEL is not set. Run 'make installed', then 'make serve MODEL=<name>')
endif
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" serve "$(MODEL)" $(ARGS)

ps: $(STAMP) ## List running servers (ps all=yes to include untracked)
	@$(LOCALHIVE) ps $(if $(filter yes,$(all)),--all,) $(ARGS)

stop: $(STAMP) ## Stop a server: make stop PORT=7000  |  make stop all=yes
ifneq ($(filter yes,$(all)),)
	@$(LOCALHIVE) stop --all
else ifneq ($(PORT),)
	@$(LOCALHIVE) stop $(PORT)
else
	$(error Give a port: 'make stop PORT=7000', or stop everything: 'make stop all=yes'. Run 'make ps' to see what is running)
endif

restart: stop ## Stop any server on PORT, then serve MODEL (PORT=, MODEL=)
ifndef MODEL
	$(error MODEL is not set. Run 'make list-models', then 'make restart MODEL=<name>')
endif
	@$(LOCALHIVE) --dir "$(MODELS_DIR)" serve "$(MODEL)" --port $(or $(PORT),7000) $(CONTENT_FLAG)

hardware: $(STAMP) ## Show detected CPU / RAM / GPU
	@$(LOCALHIVE) hardware $(ARGS)

where: ## Print the models directory
	@echo "$(MODELS_DIR)"

# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
clean-venv: ## Remove the virtualenv
	@rm -rf "$(VENV)"
	@echo "removed $(VENV)"
