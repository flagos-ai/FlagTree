#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
