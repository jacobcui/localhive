#!/usr/bin/env bash
# Verify that this machine can actually run localhive end to end.
set -uo pipefail

ok()   { printf '  \033[32m[ok]\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$*"; }

status=0

echo "localhive environment check"
echo

# --- python ---------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
  ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"
else
  bad "python3 not found"
  status=1
fi

# --- llama-server ---------------------------------------------------------
if command -v llama-server >/dev/null 2>&1; then
  ok "llama-server at $(command -v llama-server)"
  ver="$(llama-server --version 2>&1 | head -n1 || true)"
  [ -n "$ver" ] && ok "  $ver"
else
  bad "llama-server not on PATH - run 'make install'"
  status=1
fi

# --- WSL hints ------------------------------------------------------------
if grep -qi microsoft /proc/version 2>/dev/null; then
  ok "WSL2 detected"
  if [ -d /usr/lib/wsl/lib ]; then
    ok "  /usr/lib/wsl/lib present (GPU passthrough)"
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "  NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  fi
  # WSL2 caps memory via .wslconfig on the Windows side.
  mem_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
  if [ -n "$mem_kb" ]; then
    ok "  $((mem_kb / 1024 / 1024)) GiB RAM visible to WSL2"
  fi
fi

# --- localhive ------------------------------------------------------------
if python3 -c "import localhive" 2>/dev/null; then
  ok "localhive importable"
else
  warn "localhive not importable outside the venv (that is fine - use 'make')"
fi

if [ -f ".venv/bin/python" ]; then
  ok "venv present at .venv"
  if .venv/bin/python -c "import localhive" 2>/dev/null; then
    ok "  localhive importable in venv"
  else
    bad "  localhive missing from venv - run 'make deps'"
    status=1
  fi
fi

echo
if [ "$status" -eq 0 ]; then
  printf '\033[32mall good\033[0m - try: make list-models\n'
else
  printf '\033[31msome checks failed\033[0m - see above\n'
fi
exit "$status"
