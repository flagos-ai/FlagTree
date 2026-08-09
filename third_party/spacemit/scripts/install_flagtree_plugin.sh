#!/usr/bin/env bash
#
# Spacemit backend one-shot installer for FlagTree (root-triton plugin).
#
# What it does:
#   1. Idempotently apply third_party/spacemit/patch/flagtree.patch to the
#      FlagTree ROOT tree (root adaptations: CMake target-gating, LLVM22
#      dialect .td fixes, PADDING_OPTION frontend, smt/bind_sub_block merge,
#      setup.py spacemit-plugin path, setup_tools/utils/spacemit.py hook).
#      These live OUTSIDE third_party/spacemit and a pristine FlagTree checkout
#      does not have them, so they must be applied BEFORE pip runs setup.py.
#   2. Run `FLAGTREE_BACKEND=spacemit pip install .` from the FlagTree root.
#
# Usage (from anywhere):
#   bash third_party/spacemit/scripts/install_flagtree_plugin.sh
#
# Overridable env vars (verified defaults below):
#   LLVM_SYSPATH            FlagTree x86-64 LLVM (f6ded0be == LLVM22)
#   SPINE_MLIR_INSTALL_DIR  spine-mlir install (libSpeIR*.so, spine-opt, llc, ...)
#   SPINE_RUNTIME_INSTALL_DIR  spine-runtime install (libspert.so, spert headers)
#   MAX_JOBS                parallel compile jobs (default 2, prevents OOM)
#   PIP                     pip executable (default: python -m pip)
#
set -euo pipefail

# --- locate FlagTree root (this script is at third_party/spacemit/scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLAGTREE_ROOT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"
PATCH_FILE="${SPACEMIT_DIR}/patch/flagtree.patch"

echo "[spacemit] FlagTree root : ${FLAGTREE_ROOT}"
echo "[spacemit] patch file    : ${PATCH_FILE}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[spacemit] ERROR: patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

cd "${FLAGTREE_ROOT}"

# --- 1. idempotently apply flagtree.patch to the ROOT tree ---
# Detect whether the patch is already applied. `git apply -R --check` succeeds
# only when the patch is FULLY applied (reverse would work); a forward
# `--check` succeeds only when it is NOT applied yet. Use both to be safe.
#
# If the clean apply fails (context lines drifted on main), fall back to
# `git apply --3way`: it uses the blob hashes in the patch's `index` line to
# do a 3-way merge, resolving minor context drift automatically. This is
# what makes the patch resilient to FlagTree main updates — small changes
# in surrounding context no longer require a patch regen.
if git apply -R --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[spacemit] flagtree.patch already applied — skipping."
elif git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[spacemit] applying flagtree.patch ..."
  git apply "${PATCH_FILE}"
  echo "[spacemit] flagtree.patch applied."
elif git apply --3way "${PATCH_FILE}" 2>/tmp/spacemit_3way.log; then
  echo "[spacemit] flagtree.patch applied via 3-way merge (context had drifted)."
  if grep -rn '^<<<<<<<\|^=======\|^>>>>>>>' --include='*.py' --include='*.txt' \
       --include='*.td' --include='*.cc' --include='*.cmake' \
       CMakeLists.txt setup.py python/ include/ third_party/proton/ third_party/tle/ 2>/dev/null; then
    echo "[spacemit] ERROR: 3-way merge left conflict markers." >&2
    echo "           Resolve them, then re-run this script." >&2
    exit 1
  fi
else
  echo "[spacemit] ERROR: flagtree.patch does not apply cleanly (clean + 3way both failed)." >&2
  echo "           The FlagTree root has diverged beyond what 3-way merge can resolve." >&2
  echo "           Regenerate: see third_party/spacemit/patch/README.md" >&2
  echo "           3way log:" >&2
  sed 's/^/             /' /tmp/spacemit_3way.log >&2
  echo "           Inspect with: git apply --check -v ${PATCH_FILE}" >&2
  exit 1
fi

# --- 2. run the FlagTree unified install (spacemit plugin path) ---
LLVM_SYSPATH="${LLVM_SYSPATH:-/home/share/nfs_share/llvm-pre-build/llvm-f6ded0be897e2878612dd903f7e8bb85448269e5-build-x86-release}"
MAX_JOBS="${MAX_JOBS:-2}"
PIP="${PIP:-python -m pip}"

echo "[spacemit] LLVM_SYSPATH   : ${LLVM_SYSPATH}"
echo "[spacemit] MAX_JOBS       : ${MAX_JOBS}"
if [[ -n "${SPINE_MLIR_INSTALL_DIR:-}" ]]; then
  echo "[spacemit] SPINE_MLIR_INSTALL_DIR   : ${SPINE_MLIR_INSTALL_DIR}"
fi
if [[ -n "${SPINE_RUNTIME_INSTALL_DIR:-}" ]]; then
  echo "[spacemit] SPINE_RUNTIME_INSTALL_DIR: ${SPINE_RUNTIME_INSTALL_DIR}"
fi

if [[ ! -d "${LLVM_SYSPATH}/lib/cmake/llvm" ]]; then
  echo "[spacemit] ERROR: LLVM_SYSPATH invalid (no lib/cmake/llvm): ${LLVM_SYSPATH}" >&2
  exit 1
fi

# export 这两个变量, setup.py 用 os.environ.get 读取后复制对应的 .so / 头文件。
# (之前用 ${VAR:+VAR=...} 内联前缀, 但 bash 在 ${} 展开里遇到 = 会把赋值当命令执行)
export SPINE_MLIR_INSTALL_DIR="${SPINE_MLIR_INSTALL_DIR:-}"
export SPINE_RUNTIME_INSTALL_DIR="${SPINE_RUNTIME_INSTALL_DIR:-}"

FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="${LLVM_SYSPATH}" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="${MAX_JOBS}" \
${PIP} install . --no-build-isolation -v

echo "[spacemit] install finished."
