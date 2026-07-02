#!/usr/bin/env bash
#
# flagtree_build.sh — build FlagTree's Ascend backend from source.
#
# Compiles the full flagtree wheel including the TileIR dialect (TileIROps.td ->
# TileIRIR) and the TileIRToHIVM conversion. Aligned with the CI workflow
# .github/workflows/ascend-build-and-test.yml (editable, --no-build-isolation).
#
# Usage:
#   bash skill/script/flagtree_build.sh            # incremental editable build
#   CLEAN=1 bash skill/script/flagtree_build.sh    # wipe python/build first
#
# Env overrides (defaults shown):
#   PYTHON       = /root/miniconda3/envs/dlcompiler/bin/python
#   LLVM_SYSPATH = /root/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat
#   MAX_JOBS     = 32
#
set -euo pipefail

# --- repo root (this script lives in <repo>/script/) --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- configurable knobs ------------------------------------------------------
PYTHON="${PYTHON:-$(which python3)}"
export LLVM_SYSPATH="${LLVM_SYSPATH:-/root/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat}"
export MAX_JOBS="${MAX_JOBS:-32}"

# Python site-packages that provides pybind11 / nanobind cmake configs.
SP="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# --- backend + toolchain -----------------------------------------------------
# FLAGTREE_BACKEND=ascend forces FLIR_BUILD_INCUBATED=ON, which builds the
# TileIR dialect and TileIRToHIVM (see top-level CMakeLists.txt).
export FLAGTREE_BACKEND=ascend
# Use the bundled clang 21 (system clang is too old for the LLVM-21 headers).
export CC="$LLVM_SYSPATH/bin/clang"
export CXX="$LLVM_SYSPATH/bin/clang++"
export PATH="$LLVM_SYSPATH/bin:$PATH"
export LIBRARY_PATH="$LLVM_SYSPATH/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LLVM_SYSPATH/lib:${LD_LIBRARY_PATH:-}"
export TRITON_BUILD_PROTON=OFF
export TRITON_OFFLINE_BUILD=1

export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_C_COMPILER=$LLVM_SYSPATH/bin/clang \
  -DCMAKE_CXX_COMPILER=$LLVM_SYSPATH/bin/clang++ \
  -DLLVM_ENABLE_WERROR=OFF \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_CXX_FLAGS=-Wno-error=dangling-assignment-gsl \
  -DCMAKE_EXE_LINKER_FLAGS=-L$LLVM_SYSPATH/lib \
  -DCMAKE_SHARED_LINKER_FLAGS=-L$LLVM_SYSPATH/lib \
  -Dpybind11_DIR=$SP/pybind11/share/cmake/pybind11 \
  -Dnanobind_DIR=$SP/nanobind/cmake"

# --- build -------------------------------------------------------------------
cd "$REPO_ROOT/python"

if [[ "${CLEAN:-0}" == "1" ]]; then
  echo ">>> CLEAN=1: removing python/build (forces a fresh cmake configure)"
  rm -rf build
fi

echo ">>> python      : $PYTHON ($($PYTHON --version 2>&1))"
echo ">>> clang++     : $(command -v clang++) ($(clang++ --version | head -1))"
echo ">>> LLVM_SYSPATH: $LLVM_SYSPATH"
echo ">>> pybind11_DIR: $SP/pybind11/share/cmake/pybind11"
echo ">>> building flagtree (ascend, editable, --no-build-isolation) ..."

"$PYTHON" -m pip install -e . --no-build-isolation -v

echo ">>> build finished."
echo ">>> NOTE: the dlcompiler env ships its own upstream 'triton' 3.5.0 that"
echo ">>>       shadows this editable build; use a clean env to 'import triton'."
