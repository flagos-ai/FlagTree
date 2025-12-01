//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/ConvertTritonPtr/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToAddress.h"
#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
