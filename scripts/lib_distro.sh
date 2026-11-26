#!/usr/bin/env bash
# Shared OS / package-manager detection for localhive's scripts.
#
# Source this, do not execute it:
#     . "$(dirname "$0")/lib_distro.sh"
#
# It answers two questions:
#   1. what package manager does this box use?
#   2. what is the exact command to install a given capability?
#
# Capabilities are named after the underlying tool, not the distro package,
# so callers ask for "python_venv" and get the right answer everywhere.

# shellcheck shell=bash

# Path to the release metadata. Overridable so tests can point it at a
# fixture, and always quoted because an unquoted path with a space in it
# splits into two words and makes '[ -r ... ]' fail.
: "${LOCALHIVE_OS_RELEASE:=/etc/os-release}"

# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------

# Echo an id for the current distro family, used to pick command syntax.
lh_distro_id() {
  # WSL reports the distro it wraps, so this covers WSL2 for free.
  if [ -n "${LOCALHIVE_DISTRO:-}" ]; then
    printf '%s' "$LOCALHIVE_DISTRO"; return
  fi

  # An explicit override wins, so tests do not need a fake uname either.
  if [ -n "${LOCALHIVE_FORCE_LINUX:-}" ]; then
    :
  elif [ "$(uname -s)" = "Darwin" ]; then
    printf 'macos'; return
  fi

  if [ -r "$LOCALHIVE_OS_RELEASE" ]; then
    # shellcheck disable=SC1091
    . "$LOCALHIVE_OS_RELEASE"
    # Prefer ID_LIKE (e.g. "arch" for Manjaro, "fedora" for Nobara) since the
    # derivative's own ID often has no matching package syntax.
    for cand in "${ID:-}" "${ID_LIKE:-}"; do
      case "$cand" in
        debian|ubuntu) printf 'debian'; return ;;
        fedora)        printf 'fedora'; return ;;
        rhel|centos)   printf 'rhel';   return ;;
        arch)          printf 'arch';   return ;;
        opensuse*|suse) printf 'suse';  return ;;
        alpine)        printf 'alpine'; return ;;
      esac
    done
    # a couple of common ids whose ID_LIKE is unhelpful
    case "${ID:-}" in
      linuxmint|pop|kali|raspbian|zorin|elementary) printf 'debian'; return ;;
      manjaro|endeavouros|garuda|cachyos|steamos)   printf 'arch';   return ;;
      rocky|almalinux|ol|amzn)                      printf 'rhel';   return ;;
      nobara|bazzite|silverblue)                    printf 'fedora'; return ;;
    esac
    printf 'linux-unknown'; return
  fi

  printf 'unknown'
}

lh_is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]
}

lh_pretty_name() {
  if [ -n "${LOCALHIVE_PRETTY:-}" ]; then printf '%s' "$LOCALHIVE_PRETTY"; return; fi

  # Must agree with lh_distro_id: both consult the same override and the
  # same Darwin short-circuit, otherwise the error text can name one
  # platform while advising a package manager from another.
  if [ -n "${LOCALHIVE_FORCE_LINUX:-}" ]; then
    :
  elif [ "$(uname -s)" = "Darwin" ]; then
    printf 'macOS'; return
  fi

  if [ -r "$LOCALHIVE_OS_RELEASE" ]; then
    # shellcheck disable=SC1091
    . "$LOCALHIVE_OS_RELEASE"
    printf '%s' "${PRETTY_NAME:-${ID:-Linux}}"; return
  fi
  printf 'unknown'
}

# --------------------------------------------------------------------------
# Install-command hints
# --------------------------------------------------------------------------
#
# Capabilities: python_venv, python_pip, build_tools, git, cmake, curl
# Prints a single shell command line, or nothing when we cannot advise.

lh_install_hint() {
  local capability="$1"
  local distro; distro="$(lh_distro_id)"

  # Package names per capability, per distro.
  local pkgs=""
  case "$distro" in
    debian)
      case "$capability" in
        python_venv) pkgs="python3-venv python3-pip" ;;
        python_pip)  pkgs="python3-pip" ;;
        build_tools) pkgs="build-essential" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      [ -n "$pkgs" ] && printf 'sudo apt update && sudo apt install -y %s' "$pkgs"
      ;;
    fedora)
      case "$capability" in
        python_venv) pkgs="python3-pip" ;;   # venv/ensurepip ship in python3-libs
        python_pip)  pkgs="python3-pip" ;;
        build_tools) pkgs="gcc gcc-c++ make" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      [ -n "$pkgs" ] && printf 'sudo dnf install -y %s' "$pkgs"
      ;;
    rhel)
      case "$capability" in
        python_venv) pkgs="python3-pip" ;;
        python_pip)  pkgs="python3-pip" ;;
        build_tools) pkgs="gcc gcc-c++ make" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      if [ -n "$pkgs" ]; then
        if command -v dnf >/dev/null 2>&1; then
          printf 'sudo dnf install -y %s' "$pkgs"
        else
          printf 'sudo yum install -y %s' "$pkgs"
        fi
      fi
      ;;
    arch)
      case "$capability" in
        python_venv) pkgs="python" ;;
        python_pip)  pkgs="python-pip" ;;
        build_tools) pkgs="base-devel" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      [ -n "$pkgs" ] && printf 'sudo pacman -S --needed %s' "$pkgs"
      ;;
    suse)
      case "$capability" in
        python_venv) pkgs="python3-pip" ;;
        python_pip)  pkgs="python3-pip" ;;
        build_tools) pkgs="gcc gcc-c++ make" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      [ -n "$pkgs" ] && printf 'sudo zypper install -y %s' "$pkgs"
      ;;
    alpine)
      case "$capability" in
        python_venv) pkgs="python3 py3-pip" ;;
        python_pip)  pkgs="py3-pip" ;;
        build_tools) pkgs="build-base" ;;
        git)         pkgs="git" ;;
        cmake)       pkgs="cmake" ;;
        curl)        pkgs="curl" ;;
      esac
      [ -n "$pkgs" ] && printf 'sudo apk add %s' "$pkgs"
      ;;
    macos)
      case "$capability" in
        python_venv|python_pip) printf 'brew install python' ;;
        build_tools|git|cmake|curl) printf 'xcode-select --install' ;;
      esac
      ;;
    *)
      # Unknown distro: mention the package name so the user can map it.
      case "$capability" in
        python_venv|python_pip) printf '# install your distro package providing python3 venv + pip' ;;
        build_tools)            printf '# install your distro C/C++ toolchain + make' ;;
        git)                    printf '# install git' ;;
        cmake)                  printf '# install cmake' ;;
        curl)                   printf '# install curl' ;;
      esac
      ;;
  esac
}

# Print a short, distro-appropriate "how to fix this" block to stderr.
# Usage: lh_hint_block <capability> [extra line...]
lh_hint_block() {
  local capability="$1"; shift
  local hint; hint="$(lh_install_hint "$capability")"
  local distro; distro="$(lh_distro_id)"

  if [ -n "$hint" ]; then
    if [ "$distro" = "unknown" ] || [ "$distro" = "linux-unknown" ]; then
      printf '       %s (%s)\n' "$hint" "$(lh_pretty_name)" >&2
    else
      printf '           %s\n' "$hint" >&2
    fi
  fi
  local line
  for line in "$@"; do printf '       %s\n' "$line" >&2; done
}
