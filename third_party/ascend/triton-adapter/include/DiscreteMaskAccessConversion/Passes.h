/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H
#define TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H

#include "DiscreteMaskAccessConversionPass.h"

namespace mlir {
namespace triton {

/// Creates a pass to convert Triton dialect to HIVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createDiscreteMaskAccessConversionPass();

#define GEN_PASS_REGISTRATION
#include "ascend/triton-adapter/include/DiscreteMaskAccessConversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DISCRETE_MASK_ACCESS_CONVERSION_PASSES_H