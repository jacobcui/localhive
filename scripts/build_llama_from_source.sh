#!/usr/bin/env bash
# Source-build fallback for llama.cpp, used when no prebuilt release
# tarball matches this machine. Expects build essentials to be installed.
set -euo pipefail

say() { printf '\033[36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# shellcheck source=lib_distro.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib_distro.sh"

for tool in git cmake; do
  if ! have "$tool"; then
    printf '\033[31merror:\033[0m missing required tool: %s\n' "$tool" >&2
    printf '       Install the build prerequisites for %s:\n' "$(lh_pretty_name)" >&2
    lh_hint_block build_tools
    exit 1
  fi
done

if ! have cc && ! have gcc && ! have clang; then
  printf '\033[31merror:\033[0m no C/C++ compiler found on %s\n' "$(lh_pretty_name)" >&2
  lh_hint_block build_tools
  exit 1
fi

BUILD_DIR="${TMPDIR:-/tmp}/llama.cpp-build"
rm -rf "$BUILD_DIR"

say "cloning llama.cpp"
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$BUILD_DIR"

CMAKE_ARGS=(-B "$BUILD_DIR/build" -S "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
            -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON)

if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  say "CUDA GPU detected - enabling CUDA backend"
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
fi

say "configuring"
cmake "${CMAKE_ARGS[@]}"

JOBS="$(nproc 2>/dev/null || echo 4)"
say "compiling with $JOBS jobs (this takes a while)"
cmake --build "$BUILD_DIR/build" --config Release -j "$JOBS" --target llama-server llama-cli

DEST="${HOME}/.local/bin"
mkdir -p "$DEST"
for bin in llama-server llama-cli; do
  found="$(find "$BUILD_DIR/build" -type f -name "$bin" | head -n1)"
  [ -n "$found" ] && cp "$found" "$DEST/" && say "installed $bin"
done

# Shared libs live next to the binaries in a source build.
find "$BUILD_DIR/build" -name '*.so*' -exec cp -a {} "$DEST/" \; 2>/dev/null || true

say "done - $DEST/llama-server"
