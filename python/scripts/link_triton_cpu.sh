#!/bin/bash
# link_triton_cpu.sh — set up flagos-ai/triton-cpu sources for the FlagTree
# ARM64 CPU backend.
#
# The CPU backend's C++ extension layer (TritonCPU MLIR dialect + NEON/SVE2 C
# runtime + Python TLE builtins) lives in the separate flagos-ai/triton-cpu
# repository. This script:
#
#   1. Ensures third_party/triton-cpu/ exists — either clones flagos-ai/triton-cpu
#      fresh (default) or symlinks to an existing local clone via
#      --triton-cpu-path.
#   2. Creates the 12 symlinks the FlagTree CPU build expects at the standard
#      paths (include/triton/Dialect/TritonCPU, lib/Dialect/TritonCPU,
#      third_party/cpu/*, third_party/sleef,
#      third_party/cpu/language/cpu/{neon,runtime,tle_ops}.py,
#      python/triton/language/extra/cpu).
#
# All symlink targets are relative — moving the FlagTree worktree does not
# break the links. Idempotent — safe to re-run.
#
# Usage (run from FlagTree root after `git checkout triton_v3.3.x`):
#
#   bash python/scripts/link_triton_cpu.sh                           # clone fresh
#   bash python/scripts/link_triton_cpu.sh --triton-cpu-path PATH    # reuse local clone
#   bash python/scripts/link_triton_cpu.sh --triton-cpu-ref <SHA>    # pin a ref
#
# The script does not run pip install. After it succeeds, run:
#
#   FLAGTREE_BACKEND=cpu LLVM_SYSPATH=$(ls -d ~/.triton/llvm/llvm-a66376b0-ubuntu-arm64) \
#   TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF MAX_JOBS=$(nproc) \
#       pip install -e python/ --no-build-isolation -v

set -euo pipefail

TRITON_CPU_URL="https://github.com/flagos-ai/triton-cpu.git"
TRITON_CPU_REF="main"
TRITON_CPU_PATH=""

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | head -n -1
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --triton-cpu-url)  TRITON_CPU_URL="$2";  shift 2 ;;
        --triton-cpu-ref)  TRITON_CPU_REF="$2";  shift 2 ;;
        --triton-cpu-path) TRITON_CPU_PATH="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "[link_triton_cpu] unknown arg: $1" >&2; usage 1 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRITON_CPU_DIR="$ROOT/third_party/triton-cpu"

log() { echo "[link_triton_cpu] $*" >&2; }
die() { echo "[link_triton_cpu] ERROR: $*" >&2; exit 1; }

log "FlagTree root: $ROOT"

# Step 1: ensure third_party/triton-cpu exists -------------------------------
if [ -n "$TRITON_CPU_PATH" ]; then
    [ -d "$TRITON_CPU_PATH" ] || die "--triton-cpu-path $TRITON_CPU_PATH is not a directory"
    # Compute target as relative to third_party/ so the resulting symlink is
    # portable (the whole worktree can be moved without breaking links).
    REL_TARGET=$(realpath --relative-to="$(dirname "$TRITON_CPU_DIR")" "$TRITON_CPU_PATH")
    if [ -L "$TRITON_CPU_DIR" ] && [ "$(readlink "$TRITON_CPU_DIR")" = "$REL_TARGET" ]; then
        log "third_party/triton-cpu already links to $REL_TARGET; reusing"
    else
        [ -e "$TRITON_CPU_DIR" ] && [ ! -L "$TRITON_CPU_DIR" ] && \
            die "$TRITON_CPU_DIR exists and is not a symlink; remove it manually"
        [ -L "$TRITON_CPU_DIR" ] && rm "$TRITON_CPU_DIR"
        mkdir -p "$(dirname "$TRITON_CPU_DIR")"
        ln -sfn "$REL_TARGET" "$TRITON_CPU_DIR"
        log "third_party/triton-cpu -> $REL_TARGET"
    fi
elif [ -d "$TRITON_CPU_DIR/.git" ]; then
    log "third_party/triton-cpu already a git checkout; reusing"
elif [ -e "$TRITON_CPU_DIR" ] && [ ! -L "$TRITON_CPU_DIR" ]; then
    die "$TRITON_CPU_DIR exists but is not a git checkout; remove it or pass --triton-cpu-path"
else
    [ -L "$TRITON_CPU_DIR" ] && rm "$TRITON_CPU_DIR"
    log "cloning $TRITON_CPU_URL -> $TRITON_CPU_DIR"
    mkdir -p "$(dirname "$TRITON_CPU_DIR")"
    git clone --recursive "$TRITON_CPU_URL" "$TRITON_CPU_DIR"
    if [ "$TRITON_CPU_REF" != "main" ]; then
        log "checking out $TRITON_CPU_REF"
        git -C "$TRITON_CPU_DIR" checkout "$TRITON_CPU_REF"
        git -C "$TRITON_CPU_DIR" submodule update --init --recursive
    fi
fi

# Step 2: create the 12 symlinks --------------------------------------------
# Format: "link_path_relative_to_flagtree_root|target_relative_to_link_parent"
SYMLINKS=(
    "include/triton/Dialect/TritonCPU|../../../third_party/triton-cpu/include/triton/Dialect/TritonCPU"
    "lib/Dialect/TritonCPU|../../third_party/triton-cpu/lib/Dialect/TritonCPU"
    "third_party/cpu/CMakeLists.txt|../triton-cpu/third_party/cpu/CMakeLists.txt"
    "third_party/cpu/include|../triton-cpu/third_party/cpu/include"
    "third_party/cpu/lib|../triton-cpu/third_party/cpu/lib"
    "third_party/cpu/runtime|../triton-cpu/third_party/cpu/runtime"
    "third_party/cpu/triton_cpu.cc|../triton-cpu/third_party/cpu/triton_cpu.cc"
    "third_party/sleef|triton-cpu/third_party/sleef"
    "third_party/cpu/language/cpu/neon.py|../../../../third_party/triton-cpu/third_party/cpu/language/cpu/neon.py"
    "third_party/cpu/language/cpu/runtime.py|../../../../third_party/triton-cpu/third_party/cpu/language/cpu/runtime.py"
    "third_party/cpu/language/cpu/tle_ops.py|../../../../third_party/triton-cpu/third_party/cpu/language/cpu/tle_ops.py"
    # Pre-create as a symlink so setup.py's add_link_to_backends() sees an
    # existing symlink and refreshes it, instead of creating a real directory
    # and iterating into it (which can produce a cpu/cpu self-loop in some
    # corner cases).
    "python/triton/language/extra/cpu|../../../../third_party/cpu/language/cpu"
)

n_ok=0; n_created=0; n_updated=0
for entry in "${SYMLINKS[@]}"; do
    link_rel="${entry%%|*}"
    target="${entry#*|}"
    link="$ROOT/$link_rel"

    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
        n_ok=$((n_ok+1))
        continue
    fi
    if [ -L "$link" ]; then
        rm "$link"
        n_updated=$((n_updated+1))
    elif [ -e "$link" ]; then
        die "$link exists and is not a symlink; remove it manually before re-running"
    else
        n_created=$((n_created+1))
    fi
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
done

log "symlinks: ok=$n_ok created=$n_created updated=$n_updated (total ${#SYMLINKS[@]})"

# Sanity check: every symlink must resolve to a real file or directory.
for entry in "${SYMLINKS[@]}"; do
    link_rel="${entry%%|*}"
    [ -e "$ROOT/$link_rel" ] || die "symlink does not resolve: $link_rel"
done

log "done — next step:"
log "  FLAGTREE_BACKEND=cpu LLVM_SYSPATH=\$(ls -d ~/.triton/llvm/llvm-a66376b0-ubuntu-arm64) \\"
log "  TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF MAX_JOBS=\$(nproc) \\"
log "      pip install -e python/ --no-build-isolation -v"
