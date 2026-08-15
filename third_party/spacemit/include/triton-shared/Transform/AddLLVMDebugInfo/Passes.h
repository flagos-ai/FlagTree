//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef ADD_LLVM_DEBUG_INFO_TRANSFORM_PASSES_H
#define ADD_LLVM_DEBUG_INFO_TRANSFORM_PASSES_H

#include "triton-shared/Transform/AddLLVMDebugInfo/AddLLVMDebugInfo.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
