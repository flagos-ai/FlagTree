#!/usr/bin/env bash
#
# FlagTree riscv64 cross-compiled wheel (spacemit backend, per-Python version).
#
# Usage (works for CI and local):
#   bash third_party/spacemit/scripts/build_wheel_riscv64.sh \
#       <LLVM_cross> <spine-mlir-riscv-install> <spine-runtime-install> \
#       <mlir-bindings-dir> <py-riscv64-dir>
#
# Args (all directories, 相对路径会被解析为绝对路径):
#   1. LLVM cross install (f6ded0be == LLVM22 cross; llvm_installed/ in CI)
#   2. spine-mlir riscv64 install (bin/ + lib/, SPEIR 0.8.0 riscv package)
#   3. spine-runtime install (include/ + lib/, libspert.so source)
#   4. MLIR Python bindings package root (mlir/ at top level, per-CPython assets)
#   5. py-riscv64 root (include/python3.X + lib/libpython3.X.so)
#
# Overridable env vars:
#   MAX_JOBS          parallel build jobs (default 2, CI memory-limited)
#   PY_RISCV_ROOT     equivalent to arg 5 (arg takes precedence)
#   SPINE_MLIR_BINDINGS_DIR  equivalent to arg 4 (arg takes precedence)
#   DIST_DIR          wheel output directory (default dist-riscv64)
#   ZLIB_LIBDIR       directory holding the cross libz (default $RISCV_ROOT_PATH/sysroot/usr/lib)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEMIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FT="$(cd "${SPACEMIT_DIR}/../.." && pwd)"

# --- Arg parsing: relative paths are resolved via pushd/cd 前先固化为绝对路径 ---
if [ $# -lt 5 ]; then
    echo "ERROR: need 5 args: <LLVM_cross> <spine-mlir> <spine-runtime> <mlir-bindings> <py-riscv64>" >&2
    echo "Usage: bash third_party/spacemit/scripts/build_wheel_riscv64.sh <LLVM_cross> <spine-mlir> <spine-runtime> <mlir-bindings> <py-riscv64>" >&2
    exit 1
fi
LLVM_CROSS=$(cd "${1}" && pwd)
SPINE_MLIR_INSTALL_DIR=$(cd "${2}" && pwd)
SPINE_RUNTIME_INSTALL_DIR=$(cd "${3}" && pwd)
MLIR_BINDINGS_DIR=$(cd "${4}" && pwd)
PY_RISCV_ROOT=${PY_RISCV_ROOT:-$(cd "${5}" && pwd)}

MAX_JOBS=${MAX_JOBS:-2}
DIST_DIR=${DIST_DIR:-${FT}/dist-riscv64}
TC="${SPACEMIT_DIR}/cmake/linux_riscv64.toolchain.cmake"

: "${RISCV_ROOT_PATH:?RISCV_ROOT_PATH must be set (SpacemiT toolchain root)}"

echo "[spacemit] FlagTree root    : ${FT}"
echo "[spacemit] LLVM cross       : ${LLVM_CROSS}"
echo "[spacemit] spine-mlir       : ${SPINE_MLIR_INSTALL_DIR}"
echo "[spacemit] spine-runtime    : ${SPINE_RUNTIME_INSTALL_DIR}"
echo "[spacemit] mlir bindings    : ${MLIR_BINDINGS_DIR}"
echo "[spacemit] py-riscv64       : ${PY_RISCV_ROOT}"
echo "[spacemit] RISCV_ROOT_PATH  : ${RISCV_ROOT_PATH}"

# --- Pre-flight checks (failures surfaced explicitly) ---
for p in "${LLVM_CROSS}/bin/mlir-tblgen" \
         "${SPINE_MLIR_INSTALL_DIR}/bin/spine-opt" "${SPINE_MLIR_INSTALL_DIR}/bin/llc" \
         "${SPINE_MLIR_INSTALL_DIR}/bin/mlir-translate" "${SPINE_MLIR_INSTALL_DIR}/bin/opt" \
         "${SPINE_RUNTIME_INSTALL_DIR}/include/spert.hpp" \
         "${SPINE_RUNTIME_INSTALL_DIR}/lib/libspert.so" \
         "${MLIR_BINDINGS_DIR}/mlir" \
         "${PY_RISCV_ROOT}/include" \
         "${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++" \
         "${TC}"; do
    [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done
echo "[spacemit] pre-flight checks OK"

# --- Python version probe (headers dir name must contain python3.X) ---
PY_VER=$(ls "${PY_RISCV_ROOT}/include" | grep -m1 '^python3\.')
if [ -z "${PY_VER}" ]; then
    echo "ERROR: no python3.* include dir under ${PY_RISCV_ROOT}/include" >&2
    exit 1
fi
PYLIB="${PY_RISCV_ROOT}/lib/lib${PY_VER}.so"
[ -e "${PYLIB}" ] || [ -e "${PYLIB}.1.0" ] || { echo "FATAL: missing ${PYLIB}" >&2; exit 1; }
echo "[spacemit] target python    : ${PY_VER}"

# --- zlib (toolchain sysroot by default; CI pre-installs via third_party/spacemit/scripts/build_rv_zlib.sh) ---
ZLIB_LIBDIR=${ZLIB_LIBDIR:-${RISCV_ROOT_PATH}/sysroot/usr/lib}
if [ ! -e "${ZLIB_LIBDIR}/libz.a" ] && [ ! -e "${ZLIB_LIBDIR}/libz.so" ]; then
    echo "ERROR: no libz under ${ZLIB_LIBDIR}" >&2
    echo "Build zlib for riscv64 into the toolchain sysroot (third_party/spacemit/scripts/build_rv_zlib.sh) or set ZLIB_LIBDIR" >&2
    exit 1
fi

# --- Hard check: the riscv64 branch must be present in the CMakeLists
#     arch table (the old flagtree.patch-era hunk is now committed directly
#     into the root CMakeLists.txt; absence means the base tree lacks it) ---
cd "${FT}"
grep -q 'CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64"' CMakeLists.txt || {
    echo "FATAL: CMakeLists.txt lacks the riscv64 branch" >&2; exit 1; }

# --- Clean the previous x86 build cache (cross and x86 share the build dir; mixing them always breaks) ---
rm -rf build/cmake.linux-x86_64-cpython-* build/lib.linux-x86_64-cpython-* build/temp.linux-x86_64-cpython-*

# --- Cross build (pip wheel goes through the spacemit in-tree build path at the FlagTree root) ---
export RISCV_ROOT_PATH
export SPINE_MLIR_INSTALL_DIR
export SPINE_RUNTIME_INSTALL_DIR
# spacemit.py install_extension's bindings probe accepts the flat layout with mlir/ at top level (-python release package)
export SPINE_MLIR_BINDINGS_DIR="${MLIR_BINDINGS_DIR}"
export TRITON_APPEND_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${TC} \
-DPython3_INCLUDE_DIR:PATH=${PY_RISCV_ROOT}/include/${PY_VER} \
-DPython3_LIBRARY:FILEPATH=${PYLIB} \
-DPython3_FIND_STRATEGY=LOCATION \
-DPython3_FIND_REGISTRY=NEVER \
-DPython3_FIND_FRAMEWORK=NEVER \
-DCMAKE_SHARED_LINKER_FLAGS=-L${ZLIB_LIBDIR} \
-DCMAKE_EXE_LINKER_FLAGS=-L${ZLIB_LIBDIR}"
export CC="${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc"
export CXX="${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++"

echo "[spacemit] == cross build (~30-60 min) =="
FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH="${LLVM_CROSS}" \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS="${MAX_JOBS}" \
python3 -m pip wheel . --no-build-isolation -w "${DIST_DIR}"

# --- Verify the compiler is really riscv (toolchain-file config has no bare CMAKE_CXX_COMPILER: entry) ---
CMAKE_CACHE=$(ls build/cmake.linux-x86_64-cpython-*/CMakeCache.txt 2>/dev/null | head -1 || true)
if [ -z "${CMAKE_CACHE}" ]; then
    echo "[spacemit] WARNING: CMakeCache.txt not found — skipping compiler sanity check" >&2
else
    cxxline=$(grep -E '^CMAKE_CXX_COMPILER:' "${CMAKE_CACHE}" || true)
    [ -z "$cxxline" ] && cxxline=$(grep -E '^CMAKE_CXX_COMPILER_(AR|RANLIB):' "${CMAKE_CACHE}" | head -1 || true)
    echo "$cxxline"
    case "$cxxline" in
        *riscv64-unknown-linux-gnu*) echo "[spacemit] compiler OK (riscv cross)" ;;
        *) echo "[spacemit] ERROR: compiler is not a riscv cross: '$cxxline'" >&2; exit 1 ;;
    esac
fi

# --- Retag the wheel platform (pip wheel on an x86 host produces a linux_x86_64 tag) ---
whl=$(ls "${DIST_DIR}"/*-linux_x86_64.whl 2>/dev/null | head -1 || true)
if [ -n "${whl}" ]; then
    python3 -m wheel tags --platform-tag linux_riscv64 --remove "${whl}" \
        || { echo "FATAL: wheel retag failed (python3 -m wheel / wheel package available?)" >&2; exit 1; }
fi
whl=$(ls "${DIST_DIR}"/*-linux_riscv64.whl | head -1)
echo "[spacemit] done: ${whl}"
