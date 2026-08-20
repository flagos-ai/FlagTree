# Iluvatar CoreX device-side FlagCX adapter.
#
# The adapter is compiled outside the downloaded FlagCX tree and only uses
# headers/layouts from FLAGCX_ROOT. This keeps the NVIDIA device binding
# untouched and makes the target architecture an explicit build parameter.

FLAGCX_ROOT ?= $(abspath ../../tle/third_party/flagcx)
ILUVATAR_ARCH ?= ivcore11
ILUVATAR_SDK ?= /usr/local/corex
COREX_CLANG ?= clang
BUILD_DIR ?= $(FLAGCX_ROOT)/build
OBJ_DIR := $(BUILD_DIR)/obj/iluvatar_device
LIB_DIR := $(BUILD_DIR)/lib
SOURCE := $(abspath flagcx_device_comm_query.cu)
OUTPUT := $(LIB_DIR)/libflagcx_device.bc

SUPPORTED_ARCHES := ivcore11
ifeq ($(filter $(ILUVATAR_ARCH),$(SUPPORTED_ARCHES)),)
  $(error Unsupported Iluvatar FlagCX device target '$(ILUVATAR_ARCH)'; currently supported: $(SUPPORTED_ARCHES))
endif

FLAGCX_INCLUDE_DIRS := \
  -I$(FLAGCX_ROOT)/flagcx/adaptor/include \
  -I$(FLAGCX_ROOT)/flagcx/include

CLANG_FLAGS := \
  -x ivcore \
  --cuda-path=$(ILUVATAR_SDK) \
  --cuda-device-only \
  --cuda-gpu-arch=$(ILUVATAR_ARCH) \
  -D__clang_llvm_bitcode_lib__ \
  -O1 \
  -emit-llvm \
  -c \
  $(FLAGCX_INCLUDE_DIRS)

.PHONY: all bitcode check-symbols clean

all: bitcode check-symbols

bitcode: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	@mkdir -p $(OBJ_DIR) $(LIB_DIR)
	@echo "[FlagCX][Iluvatar] Compiling $(SOURCE)"
	$(COREX_CLANG) $(CLANG_FLAGS) $(SOURCE) -o $@

check-symbols: $(OUTPUT)
	@set -eu; \
	for symbol in \
	  flagcxDevCommGetIntraRank \
	  flagcxDevCommGetIntraSize \
	  flagcxIntraBarrierArriveS \
	  flagcxIntraBarrierWaitS \
	  flagcxIntraBarrierSyncS \
	  flagcxGetIntraPointerC; do \
	  if ! llvm-nm "$(OUTPUT)" | awk '{print $$3}' | grep -qx "$$symbol"; then \
	    echo "[FlagCX][Iluvatar] missing device symbol: $$symbol" >&2; \
	    exit 1; \
	  fi; \
	done
	@echo "[FlagCX][Iluvatar] Required TLE device symbols verified"

clean:
	rm -f $(OUTPUT)
