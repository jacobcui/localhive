# Configuration for llama model serving
# Override these values in config.local.mk (gitignored)

# Model parameters
GPU_LAYERS ?= 70
CTX_SIZE ?= 131072
MAX_TOKENS ?= 8192

# Server parameters
PORT ?= 8000

# To override, create config.local.mk with your values:
#   GPU_LAYERS ?= 90
#   PORT ?= 9000
