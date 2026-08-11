//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
