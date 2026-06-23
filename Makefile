# Makefile for managing llama models
# Helper for running llama app (@/home/jacob/miniforge3/bin/llama)
# Models are stored in ~/models directory

LLAMA := /home/jacob/miniforge3/bin/llama
MODELS_DIR := $(HOME)/models

.DEFAULT_GOAL := help

.PHONY: list serve help

help:
	@echo "Available targets:"
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
	$(LLAMA) serve -m $(MODELS_DIR)/$(MODEL) -ngl 999 -c 131072 -n 8192

serve-from-list:
	@echo "Available models:"
	@i=1; for m in $$(ls -1 $(MODELS_DIR) 2>/dev/null); do echo "  $$i) $$m"; i=$$((i+1)); done
	@echo ""
	@read -p "Select model (1-$$((i-1))): " choice; \
	if [ -z "$$choice" ]; then echo "No selection made"; exit 1; fi; \
	model=$$(ls -1 $(MODELS_DIR) 2>/dev/null | sed -n "$${choice}p"); \
	if [ -z "$$model" ]; then echo "Invalid selection: $$choice"; exit 1; fi; \
	echo "Starting llama server with model: $$model"; \
	$(LLAMA) serve -m $(MODELS_DIR)/$$model -ngl 999 -c 131072 -n 8192
