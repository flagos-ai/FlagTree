#!/bin/bash
set -euo pipefail


YELLOW='\033[33m'
GREEN='\033[32m'
RED='\033[31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTREE_ROOT="$(realpath "${SCRIPT_DIR}/..")"
THIRD_PARTY_DIR="${FLAGTREE_ROOT}/third_party/tle/third_party"
FLAGCX_DIR="${THIRD_PARTY_DIR}/flagcx"
CACHE_DIR="${HOME}/.flagtree/flagcx"

SO_FILE="${FLAGCX_DIR}/build/lib/libflagcx.so"
BC_FILE="${FLAGCX_DIR}/build/lib/libflagcx_device.bc"

mkdir -p "${THIRD_PARTY_DIR}"
mkdir -p "${CACHE_DIR}"

echo -e "[INFO] FlagTree root : ${FLAGTREE_ROOT}"
echo -e "[INFO] FlagCX dir    : ${FLAGCX_DIR}"
echo -e "[INFO] Cache dir     : ${CACHE_DIR}$"


#
# Clone FlagCX
#
if [ ! -d "${FLAGCX_DIR}" ]; then
    git clone https://github.com/flagos-ai/FlagCX.git "${FLAGCX_DIR}"
    cd "${FLAGCX_DIR}"
else
    echo -e "${GREEN}[INFO] FlagCX already exists${NC}"
fi

pushd "${FLAGCX_DIR}"



#
# libflagcx.so
#
if [[ -f "${SO_FILE}" ]]; then
    echo "[INFO] libflagcx.so already exists, skip."
else
    echo -e "${YELLOW}[Compiling] Building libflagcx.so ...${NC}"
    make USE_NVIDIA=1 -j"$(nproc)"

    [[ -f "${SO_FILE}" ]] || {
        echo "[ERROR] Failed to generate ${SO_FILE}"
        exit 1
    }
fi


#
# libflagcx_device.bc
#
if [[ -f "${BC_FILE}" ]]; then
    echo "[INFO] libflagcx_device.bc already exists, skip."
else
    echo -e "${YELLOW}[Compiling] Building libflagcx_device.bc ...${NC}"
    make -C bindings/ir/nvidia

    [[ -f "${BC_FILE}" ]] || {
        echo "[ERROR] Failed to generate ${BC_FILE}"
        exit 1
    }
fi


echo -e "${GREEN}[DONE] Build finished${NC}"


echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/libflagcx.so -> ${CACHE_DIR}${NC}"
cp build/lib/libflagcx.so "${CACHE_DIR}"

echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/libflagcx_device.bc -> ${CACHE_DIR}${NC}"
cp build/lib/libflagcx_device.bc "${CACHE_DIR}"

echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/flagcx_wrapper.py -> ${CACHE_DIR}${NC}"
cp plugin/interservice/flagcx_wrapper.py "${CACHE_DIR}"

echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/include/ -> ${CACHE_DIR}${NC}"
cp -r flagcx/include "${CACHE_DIR}"

# wrapper
echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/plugin/interservice/flagcx_wrapper.py -> ${FLAGTREE_ROOT}/python/triton/experimental/tle/language/${NC}"
cp plugin/interservice/flagcx_wrapper.py \
   "${FLAGTREE_ROOT}/python/triton/experimental/tle/language/"

echo -e "${YELLOW}[Copying] ${FLAGCX_DIR}/include/ -> ${FLAGTREE_ROOT}/python/triton/experimental/tle/language/include/${NC}"
cp -r flagcx/include \
      "${FLAGTREE_ROOT}/python/triton/experimental/tle/language/include"


echo -e "${GREEN}[DONE] FlagCX setup completed. ${NC}"
printf "\n\n"
popd
