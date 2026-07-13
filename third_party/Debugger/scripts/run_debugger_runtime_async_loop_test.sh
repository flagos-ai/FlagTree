#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${FLAGTREE_DEBUGGER_RUNTIME_BUILD_DIR:-/tmp/flagtree-debugger-runtime-build}"

echo "[runner] root=${ROOT_DIR}"
echo "[runner] build_dir=${BUILD_DIR}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  echo "[runner] sourcing Ascend environment"
  had_nounset=0
  if [[ $- == *u* ]]; then
    had_nounset=1
    set +u
  fi
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  if [[ ${had_nounset} -eq 1 ]]; then
    set -u
  fi
fi

echo "[runner] configuring standalone runtime build"
cmake -S "${ROOT_DIR}/lib/Runtime" -B "${BUILD_DIR}"
echo "[runner] building FlagTreeDebuggerRuntimeAsyncLoopTest"
cmake --build "${BUILD_DIR}" --target FlagTreeDebuggerRuntimeAsyncLoopTest -j 4

echo "[runner] running async loop test"
exec env \
  FLAGTREE_DEBUGGER_RUNTIME_ASYNC_DEVICE="${FLAGTREE_DEBUGGER_RUNTIME_ASYNC_DEVICE:-0}" \
  FLAGTREE_DEBUGGER_RUNTIME_ASYNC_ITERS="${FLAGTREE_DEBUGGER_RUNTIME_ASYNC_ITERS:-32}" \
  FLAGTREE_DEBUGGER_RUNTIME_ASYNC_RECORDS="${FLAGTREE_DEBUGGER_RUNTIME_ASYNC_RECORDS:-4}" \
  "${BUILD_DIR}/FlagTreeDebuggerRuntimeAsyncLoopTest"
