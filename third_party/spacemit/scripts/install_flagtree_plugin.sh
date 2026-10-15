#!/usr/bin/env bash
#
# Spacemit backend one-shot installer for FlagTree (root-triton plugin).
#
# What it does:
#   Run `FLAGTREE_BACKEND=spacemit pip install .` from the FlagTree root.
#
# Usage (from anywhere):
#   bash third_party/spacemit/scripts/install_flagtree_plugin.sh
#
# Overridable env vars:
#   LLVM_SYSPATH            FlagTree x86-64 LLVM (f6ded0be == LLVM22)
#   SPINE_MLIR_INSTALL_DIR  spine-mlir install (libSpeIR*.so, spine-opt, llc, ...)
#   SPINE_RUNTIME_INSTALL_DIR  spine-runtime install (libspert.so, spert headers)
# If these paths are unset, setup_tools/utils/spacemit.py downloads the assets
# described by the *_URL and *_MD5 variables in spacemit-ci.env.
#   MAX_JOBS                parallel compile jobs (default 2, prevents OOM)
#   PIP                     pip executable (default: python -m pip)
#
set -euo pipefail

# --- locate FlagTree root (this script is at third_party/spacemit/scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLAGTREE_ROOT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"

echo "[spacemit] FlagTree root : ${FLAGTREE_ROOT}"
cd "${FLAGTREE_ROOT}"

# --- Run the FlagTree unified install (spacemit plugin path) ---
# Preserve the local NFS fallback, but do not let it bypass the CI cache when
# spacemit-ci.env has supplied an LLVM download.
MAX_JOBS="${MAX_JOBS:-32}"
PIP="${PIP:-python3 -m pip}"
export PIP_BREAK_SYSTEM_PACKAGES=1

if [[ -n "${LLVM_SYSPATH:-}" ]]; then
  echo "[spacemit] LLVM_SYSPATH   : ${LLVM_SYSPATH}"
else
  echo "[spacemit] LLVM_SYSPATH   : managed by register_cache"
fi
echo "[spacemit] MAX_JOBS       : ${MAX_JOBS}"
if [[ -n "${SPINE_MLIR_INSTALL_DIR:-}" ]]; then
  echo "[spacemit] SPINE_MLIR_INSTALL_DIR   : ${SPINE_MLIR_INSTALL_DIR}"
fi
if [[ -n "${SPINE_RUNTIME_INSTALL_DIR:-}" ]]; then
  echo "[spacemit] SPINE_RUNTIME_INSTALL_DIR: ${SPINE_RUNTIME_INSTALL_DIR}"
fi

if [[ -n "${LLVM_SYSPATH:-}" && ! -d "${LLVM_SYSPATH}/lib/cmake/llvm" ]]; then
  echo "[spacemit] ERROR: LLVM_SYSPATH invalid (no lib/cmake/llvm): ${LLVM_SYSPATH}" >&2
  exit 1
fi

# export 这两个变量, setup.py 用 os.environ.get 读取后复制对应的 .so / 头文件。
# (之前用 ${VAR:+VAR=...} 内联前缀, 但 bash 在 ${} 展开里遇到 = 会把赋值当命令执行)
export SPINE_MLIR_INSTALL_DIR="${SPINE_MLIR_INSTALL_DIR:-}"
export SPINE_RUNTIME_INSTALL_DIR="${SPINE_RUNTIME_INSTALL_DIR:-}"

FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="${LLVM_SYSPATH:-}" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="${MAX_JOBS}" \
${PIP} install . --no-build-isolation -v

echo "[spacemit] install finished."
