/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_TRITON_TO_ANNOTATION_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_ANNOTATION_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to Annotation dialect.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToAnnotationPass();

#define GEN_PASS_REGISTRATION
#include "ascend/triton-adapter/include/TritonToAnnotation/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_ANNOTATION_CONVERSION_PASSES_H