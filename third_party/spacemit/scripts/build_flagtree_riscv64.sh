#!/usr/bin/env bash
#
# FlagTree riscv64 cross-compiled wheel (spacemit backend, 2026-09-29)
#
# What it does:
#   0. Pre-flight checks: 所有依赖路径必须存在
#   1. Pin the riscv spine-mlir tools into .riscv-assets/ (guard against ci-builds rotation cleanup)
#   2. Assemble the riscv spine-runtime 0.6.3 (libspert + include)
#   3. Back up and clean the x86 build cache / backend binaries (hard-check the riscv64 CMakeLists branch)
#   4. pip wheel cross build (~30-60 min)
#   5. Verify CMAKE_CXX_COMPILER is really the riscv cross compiler
#   6. Retag the wheel platform tag to linux_riscv64 + verify contents
#   7. Restore the source tree to x86 state
#
# Usage (from anywhere):
#   bash third_party/spacemit/scripts/build_flagtree_riscv64.sh 2>&1 | tee ~/tmp/ft_riscv_build.log
#
# Overridable env vars (defaults = local NFS / CI runner workspace layout):
#   W                  CI runner workspace (holds .venv-ci / llvm_installed / rpc_runtime_installed /
#                      spine_mlir_installed), default <FlagTree parent>/github-ci/actions-runner/_work/FlagTree/FlagTree
#   LLVM_CROSS         riscv64 cross-compile LLVM (f6ded0be)
#   SML_SRC            riscv spine-mlir installed source (ci-builds artifact, pinned into .riscv-assets by step 1)
#   SPINE_RUNTIME_SRC  spine-runtime source repo (for include/)
#   TOOLCHAIN_ROOT     SpacemiT cross toolchain (exported as RISCV_ROOT_PATH)
#   PYINC              riscv Python headers dir (for cross-compiling Python.h)
#   ZLIB_LIBDIR        directory holding the static cross libz.a
#   MAX_JOBS           parallel build jobs (default 64)
#   TMPDIR             build temp dir (default $HOME/tmp)
#
set -euo pipefail

# --- Self-locate the FlagTree root (this script lives in third_party/spacemit/scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${FT}/.." && pwd)"

# --- External dependencies (all overridable via env) ---
W="${W:-$WORKSPACE/github-ci/actions-runner/_work/FlagTree/FlagTree}"
LLVM_CROSS="${LLVM_CROSS:-/share/nfs_share/llvm-pre-build/llvm-f6ded0be897e2878612dd903f7e8bb85448269e5-20260928-release-cross}"
SML_SRC="${SML_SRC:-/share/nfs_share/ci-builds/ai/AICompiler/spine-mlir/24462/build-riscv64/installed}"
SPINE_RUNTIME_SRC="${SPINE_RUNTIME_SRC:-$WORKSPACE/spine-runtime}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/share/nfs_share/ci-node-compiler-env/spacemit-toolchain-linux-glibc-x86_64-v1.1.2}"
PYINC="${PYINC:-/share/nfs_share/env/python3.12.7/include/python3.12}"
ZLIB_LIBDIR="${ZLIB_LIBDIR:-$HOME/work/riscv-deps/zlib/lib}"
MAX_JOBS="${MAX_JOBS:-64}"
TMPDIR="${TMPDIR:-$HOME/tmp}"

# --- Derived paths ---
SML="$FT/.riscv-assets/spine_mlir_installed"
RT="$FT/.riscv-assets/spine_runtime_installed"
TC="$SPACEMIT_DIR/cmake/linux_riscv64.toolchain.cmake"

echo "[spacemit] FlagTree root  : $FT"
echo "[spacemit] CI workspace   : $W"
echo "[spacemit] LLVM cross     : $LLVM_CROSS"
echo "[spacemit] spine-mlir src : $SML_SRC"
echo "[spacemit] toolchain      : $TOOLCHAIN_ROOT"

echo "[spacemit] == [0/7] pre-flight checks: 所有依赖路径必须存在 =="
for p in "$LLVM_CROSS/bin/mlir-tblgen" "$SML_SRC/bin/spine-opt" "$SML_SRC/bin/llc" \
         "$SML_SRC/bin/mlir-translate" "$SML_SRC/bin/opt" "$TC" \
         "$W/rpc_runtime_installed/lib/libspert.so.0.6.3" \
         "$SPINE_RUNTIME_SRC/include/spert.hpp" \
         "$TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-g++" "$PYINC/Python.h" \
         "$W/.venv-ci/bin/activate" "$W/llvm_installed/bin" \
         "$W/spine_mlir_installed/bin/spine-opt" "$ZLIB_LIBDIR"; do
  [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done
echo "[spacemit] pre-flight checks OK"

echo "[spacemit] == [1/7] pin riscv spine-mlir tools (guard against ci-builds rotation cleanup) =="
mkdir -p "$FT/.riscv-assets"
[ -d "$SML/bin" ] || cp -a "$SML_SRC" "$SML"

echo "[spacemit] == [2/7] assemble riscv spine-runtime 0.6.3 =="
mkdir -p "$RT/lib"
cp -a "$W/rpc_runtime_installed/lib/libspert.so"* "$RT/lib/"
[ -d "$RT/include" ] || cp -a "$SPINE_RUNTIME_SRC/include" "$RT/"

echo "[spacemit] == [3/7] back up and clean the x86 build cache / backend binaries =="
cd "$FT"
# Hard check: the riscv64 branch must be present in the CMakeLists arch
# table (the old flagtree.patch-era hunk is now committed directly into the
# root CMakeLists.txt)
grep -q 'CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64"' CMakeLists.txt || {
  echo "FATAL: CMakeLists.txt lacks the riscv64 branch" >&2; exit 1; }
echo "[spacemit] riscv64 CMakeLists branch in place"

mkdir -p .riscv-assets/backup-backend
[ -d .riscv-assets/backup-backend/bin ] || cp -a third_party/spacemit/backend/bin .riscv-assets/backup-backend/bin
[ -d .riscv-assets/backup-backend/lib ] || cp -a third_party/spacemit/backend/lib .riscv-assets/backup-backend/lib
rm -rf build/cmake.linux-x86_64-cpython-3.12 build/lib.linux-x86_64-cpython-312 build/temp.linux-x86_64-cpython-312
rm -f third_party/spacemit/backend/bin/llc third_party/spacemit/backend/bin/mlir-translate \
      third_party/spacemit/backend/bin/opt third_party/spacemit/backend/bin/spine-opt \
      third_party/spacemit/backend/bin/spine-triton-opt
rm -f third_party/spacemit/backend/lib/libspert* third_party/spacemit/backend/lib/libSpeIR*

echo "[spacemit] == [4/7] cross build (~30-60 min) =="
echo "    (after ~1 min, sanity-check in another terminal: grep CMAKE_CXX_COMPILER_ $FT/build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt"
echo "      the path must contain riscv64-unknown-linux-gnu; if it is /usr/bin, Ctrl-C immediately)"
mkdir -p "$TMPDIR"
source "$W/.venv-ci/bin/activate"
export RISCV_ROOT_PATH="$TOOLCHAIN_ROOT"
export TMPDIR
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$TC -DPython3_INCLUDE_DIR=$PYINC -DCMAKE_SHARED_LINKER_FLAGS=-L$ZLIB_LIBDIR -DCMAKE_EXE_LINKER_FLAGS=-L$ZLIB_LIBDIR"
echo "[spacemit] TRITON_APPEND_CMAKE_ARGS=$TRITON_APPEND_CMAKE_ARGS"

FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="$LLVM_CROSS" \
SPINE_MLIR_INSTALL_DIR="$SML" \
SPINE_RUNTIME_INSTALL_DIR="$RT" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="$MAX_JOBS" \
pip wheel . --no-build-isolation -w dist-riscv64

echo "[spacemit] == [5/7] verify the compiler is really riscv =="
# Note: a toolchain-file-configured CMakeCache has no bare
# CMAKE_CXX_COMPILER: entry (only CMAKE_CXX_COMPILER_AR/RANLIB); grepping
# directly would miss and silently exit under set -e.
cxxline=$(grep -E '^CMAKE_CXX_COMPILER:' build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt || true)
[ -z "$cxxline" ] && cxxline=$(grep -E '^CMAKE_CXX_COMPILER_(AR|RANLIB):' build/cmake.linux-x86_64-cpython-3.12/CMakeCache.txt | head -1 || true)
echo "$cxxline"
case "$cxxline" in
  *riscv64-unknown-linux-gnu*) echo "[spacemit] compiler OK (riscv cross)" ;;
  *) echo "[spacemit] ERROR: compiler is not a riscv cross: '$cxxline'" >&2; exit 1 ;;
esac

echo "[spacemit] == [6/7] retag platform tag + verify wheel contents =="
whl=$(ls dist-riscv64/*-linux_x86_64.whl)
python3 -m wheel tags --platform-tag linux_riscv64 --remove "$whl"
whl=$(ls dist-riscv64/*-linux_riscv64.whl)
chk="$TMPDIR/whlchk"
rm -rf "$chk"; mkdir -p "$chk"
unzip -o -q "$FT/$whl" -d "$chk" 'triton/_C/libtriton*' 'triton/backends/spacemit/bin/*' 'triton/backends/spacemit/lib/*'
so=$(find "$chk" -name 'libtriton*.so' | head -1)
echo "-- libtriton.so: $(readelf -h "$so" | grep Machine)"
echo "-- backend/bin:"; file "$chk"/triton/backends/spacemit/bin/* | grep -v __init__ || true
echo "-- backend/lib:"; ls -la "$chk/triton/backends/spacemit/lib/"

echo "[spacemit] == [7/7] restore the source tree to x86 state =="
rm -rf third_party/spacemit/backend/bin third_party/spacemit/backend/lib
cp -a .riscv-assets/backup-backend/bin third_party/spacemit/backend/bin
cp -a .riscv-assets/backup-backend/lib third_party/spacemit/backend/lib
cp "$W/spine_mlir_installed/bin/spine-opt" third_party/spacemit/backend/bin/spine-opt
ln -sfn "$W/llvm_installed" "$HOME/.triton/llvm/llvm-ubuntu-x64"
echo "[spacemit] done: $FT/$whl"
