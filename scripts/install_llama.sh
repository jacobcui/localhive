#!/usr/bin/env bash
# Install llama.cpp for localhive.
#
#   macOS          -> Homebrew (brew install llama.cpp)
#   Linux / WSL2   -> official prebuilt release, CPU or CUDA flavour,
#                     falling back to a source build when no tarball fits.
#
# WSL2 specifics we handle:
#   * the GPU is only visible if /usr/lib/wsl/lib is on the library path
#   * CUDA is exposed through the Windows driver, so we must NOT install a
#     Linux NVIDIA driver inside the distro
set -euo pipefail

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# shellcheck source=lib_distro.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib_distro.sh"
is_wsl() { lh_is_wsl; }

# --- already good? ---------------------------------------------------------
if have llama-server; then
  say "llama-server already present: $(command -v llama-server)"
  llama-server --version 2>&1 | head -n 2 || true
  exit 0
fi

# --- macOS -----------------------------------------------------------------
if [ "$(uname -s)" = "Darwin" ]; then
  have brew || die "Homebrew is required on macOS: https://brew.sh"
  say "installing llama.cpp via Homebrew"
  brew install llama.cpp
  say "installed: $(command -v llama-server)"
  exit 0
fi

# --- Linux / WSL2 ----------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "unsupported platform: $(uname -s)"

if is_wsl; then
  say "WSL2 detected (${WSL_DISTRO_NAME:-unknown distro})"
  # The Windows driver provides libcuda.so here; keep it on the path.
  if [ -d /usr/lib/wsl/lib ] && ! echo "${LD_LIBRARY_PATH:-}" | grep -q /usr/lib/wsl/lib; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
  fi
  if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    say "NVIDIA GPU visible through WSL: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  else
    warn "no NVIDIA GPU visible; will install the CPU build"
  fi
fi

# Detect an NVIDIA GPU to pick the CUDA build.
GPU_KIND="cpu"
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  GPU_KIND="cuda"
fi

# llama.cpp publishes prebuilt Linux tarballs linked against a specific
# Ubuntu release. They generally run on any glibc distro new enough to
# satisfy the link, so we pick the *oldest* ABI that still works rather than
# guessing, and fall back to a source build when the distro is clearly not
# glibc-compatible.
CODENAME=""
DISTRO="$(lh_distro_id)"

case "$DISTRO" in
  debian)
    # Prefer the distro's own codename when llama.cpp ships a matching build.
    if [ -r "$LOCALHIVE_OS_RELEASE" ]; then
      # shellcheck disable=SC1091
      . "$LOCALHIVE_OS_RELEASE"
      case "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}" in
        22.04|jammy)    CODENAME="ubuntu-22.04" ;;
        24.04|noble)    CODENAME="ubuntu-24.04" ;;
        26.04|resolute) CODENAME="ubuntu-24.04" ;;  # 26.04 keeps the 24.04 ABI
        *)              CODENAME="" ;;
      esac
    fi
    # Debian proper, Mint, Pop, Kali etc. are not Ubuntu; the 22.04 build is
    # the safest glibc floor and is what upstream effectively targets.
    [ -n "$CODENAME" ] || CODENAME="ubuntu-22.04"
    ;;
  fedora|rhel|suse|arch)
    # glibc is new enough for the 24.04 build on all of these.
    CODENAME="ubuntu-24.04"
    say "note: $DISTRO is not Ubuntu; using the $CODENAME build (glibc-compatible)"
    ;;
  alpine)
    # musl, not glibc: upstream tarballs will not run. Build from source.
    warn "Alpine uses musl, so the prebuilt glibc tarballs will not run"
    say "building from source instead"
    bash "$(dirname "$0")/build_llama_from_source.sh"
    exit $?
    ;;
  *)
    warn "unrecognised Linux distribution: $(lh_pretty_name)"
    if have ldd && ldd --version 2>&1 | grep -qi musl; then
      say "musl libc detected; building from source"
      bash "$(dirname "$0")/build_llama_from_source.sh"
      exit $?
    fi
    CODENAME="ubuntu-22.04"
    say "assuming glibc; trying the $CODENAME build"
    ;;
esac

say "release flavour: $CODENAME / $GPU_KIND"

DEST="${HOME}/.local/bin"
mkdir -p "$DEST"

arch="$(uname -m)"
case "$arch" in
  x86_64|amd64) asset_arch="x64" ;;
  aarch64|arm64) asset_arch="arm64" ;;
  *) die "unsupported architecture: $arch" ;;
esac

if [ "$GPU_KIND" = "cuda" ]; then
  asset="llama-${asset_arch}-${CODENAME}-cuda"
else
  asset="llama-${asset_arch}-${CODENAME}"
fi

API="https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
say "resolving latest llama.cpp release"
if have curl; then
  release_json="$(curl -fsSL "$API" 2>/dev/null || true)"
elif have wget; then
  release_json="$(wget -qO- "$API" 2>/dev/null || true)"
else
  die "need curl or wget"
fi

url=""
if [ -n "$release_json" ] && have python3; then
  url="$(printf '%s' "$release_json" | python3 -c '
import json,sys,re
asset=sys.argv[1]
try:
    data=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for a in data.get("assets", []):
    n=a.get("name","")
    if asset in n and n.endswith(".tar.gz"):
        print(a["browser_download_url"]); break
' "$asset")"
fi

if [ -z "$url" ]; then
  warn "could not find a prebuilt '$asset' tarball; building from source instead"
  bash "$(dirname "$0")/build_llama_from_source.sh"
  exit $?
fi

say "downloading $(basename "$url")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if have curl; then
  curl -fL --progress-bar -o "$tmp/llama.tar.gz" "$url"
else
  wget -q --show-progress -O "$tmp/llama.tar.gz" "$url"
fi

say "extracting into $DEST"
tar -xzf "$tmp/llama.tar.gz" -C "$tmp"
# The tarball layout changes between releases; find the binaries wherever they are.
found="$(find "$tmp" -type f -name 'llama-server' | head -n1)"
[ -n "$found" ] || die "llama-server not found inside the downloaded archive"
bindir="$(dirname "$found")"
cp -a "$bindir"/. "$DEST"/ 2>/dev/null || cp "$bindir"/* "$DEST"/
chmod +x "$DEST"/llama-* 2>/dev/null || true

# Make sure ~/.local/bin is on PATH for future shells.
case ":$PATH:" in
  *":$DEST:"*) ;;
  *)
    for rc in "$HOME/.bashrc" "$HOME/.profile"; do
      if [ -f "$rc" ] && ! grep -q '.local/bin' "$rc"; then
        printf '\n# added by localhive\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$rc"
        say "added $DEST to PATH in $rc"
      fi
    done
    export PATH="$DEST:$PATH"
    ;;
esac

say "installed: $(command -v llama-server || echo "$DEST/llama-server")"
"$DEST/llama-server" --version 2>&1 | head -n 2 || true
say "done"
