//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/UnstructuredToMK/UnstructuredToMK.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/UnstructuredToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
