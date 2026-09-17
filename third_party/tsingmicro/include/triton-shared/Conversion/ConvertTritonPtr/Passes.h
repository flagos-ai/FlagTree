//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TRITON_PTR_CONVERSION_PASSES_H
#define CONVERT_TRITON_PTR_CONVERSION_PASSES_H

#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToAddress.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
