#!/usr/bin/env bash
# Create (or repair) the project virtualenv and install localhive into it.
#
# Idempotent: safe to run on every 'make', does nothing when already set up.
#
# Why this is a script and not three Makefile lines
# ------------------------------------------------
# On Debian/Ubuntu, 'python3 -m venv' can succeed while producing a venv with
# NO pip, if python3-venv / ensurepip is missing. That is common on a fresh
# WSL2 image and on minimal cloud images. The old Makefile guarded on
# '.venv/bin/python' existing, so it considered such a venv healthy and then
# died with:
#
#   /bin/bash: line 1: .venv/bin/pip: No such file or directory
#
# This script verifies each step actually produced what it claims to, and
# repairs or explains rather than proceeding into a confusing failure.
set -euo pipefail

VENV="${VENV:-.venv}"
STAMP="$VENV/.localhive-installed"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------ which python to build from
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for cand in python3 python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$cand"; break; fi
  done
fi
[ -n "$PYTHON_BIN" ] || die "no python3 found. Install it (e.g. sudo apt install python3 python3-venv)"

# ------------------------------------------------ sanity-check an existing venv
venv_is_broken() {
  # Missing python, missing pip, or python cannot actually run.
  [ -x "$PY" ] || return 0
  "$PY" -c "import sys" >/dev/null 2>&1 || return 0
  [ -x "$PIP" ] || return 0
  return 1
}

if [ -d "$VENV" ] && venv_is_broken; then
  if [ -x "$PY" ] && ! [ -x "$PIP" ]; then
    warn "existing $VENV has no pip (common on Debian/Ubuntu images)"
  else
    warn "existing $VENV looks broken; recreating it"
  fi
  rm -rf "$VENV"
fi

# ------------------------------------------------ create if absent
if [ ! -x "$PY" ]; then
  say "creating virtualenv at $VENV (using $PYTHON_BIN)"
  if ! "$PYTHON_BIN" -m venv "$VENV" 2>/tmp/localhive-venv-err.$$; then
    cat /tmp/localhive-venv-err.$$ >&2 || true
    rm -f /tmp/localhive-venv-err.$$
    die "could not create a virtualenv.
       On Debian/Ubuntu/WSL2 this usually means python3-venv is missing:
           sudo apt update && sudo apt install -y python3-venv python3-pip
       Then re-run 'make'."
  fi
  rm -f /tmp/localhive-venv-err.$$
fi

# ------------------------------------------------ ensure pip exists
if [ ! -x "$PIP" ]; then
  warn "venv was created without pip; bootstrapping it"
  if "$PY" -m ensurepip --upgrade >/dev/null 2>&1 && [ -x "$PIP" ]; then
    say "pip bootstrapped via ensurepip"
  else
    # No ensurepip: try to borrow pip from the system interpreter.
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
      say "installing pip from the system interpreter"
      "$PYTHON_BIN" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
      # 'pip install pip' targets the system env; copy it into the venv.
      SYS_PIP="$("$PYTHON_BIN" -c 'import pip,os;print(os.path.dirname(pip.__file__))' 2>/dev/null || true)"
      if [ -n "$SYS_PIP" ] && [ -d "$SYS_PIP" ]; then
        SITE="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
        mkdir -p "$SITE"
        cp -r "$SYS_PIP" "$SITE/" 2>/dev/null || true
        "$PY" -m pip --version >/dev/null 2>&1 && say "pip copied from system"
      fi
    fi
  fi
fi

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  die "the virtualenv at $VENV has no working pip and it could not be bootstrapped.
       Install it and re-run 'make':
           sudo apt update && sudo apt install -y python3-venv python3-pip
       Then:  rm -rf $VENV && make"
fi

# ------------------------------------------------ install the package
NEEDS_INSTALL=0
[ -f "$STAMP" ] || NEEDS_INSTALL=1
if [ -f "$STAMP" ] && [ -f pyproject.toml ] && [ pyproject.toml -nt "$STAMP" ]; then
  NEEDS_INSTALL=1
fi
if ! "$PY" -c "import localhive" >/dev/null 2>&1; then
  NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
  say "installing python deps"
  "$PIP" install --quiet --upgrade pip || \
    die "could not upgrade pip. Check your network/proxy, then re-run 'make'."

  if ! "$PIP" install --quiet -e . ; then
    die "installing localhive failed.
       Common causes:
         * no network access (corporate proxy? try: export HTTPS_PROXY=...)
         * a system-wide pip restriction (PEP 668); a venv should avoid this,
           but if it persists run: rm -rf $VENV && make
       Re-run 'make doctor-deps' for a diagnosis."
  fi

  touch "$STAMP"
  say "done"
fi

# ------------------------------------------------ verify
"$PY" -c "import localhive" >/dev/null 2>&1 || \
  die "localhive installed but not importable. Try: rm -rf $VENV && make"
[ -x "$VENV/bin/localhive" ] || \
  die "the 'localhive' entry point was not created. Try: rm -rf $VENV && make"
