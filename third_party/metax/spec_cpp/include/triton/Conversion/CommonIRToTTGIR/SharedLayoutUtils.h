#ifndef TRITON_METAX_COMMONIR_SHARED_LAYOUT_UTILS_H
#define TRITON_METAX_COMMONIR_SHARED_LAYOUT_UTILS_H

#include "mlir-ext/Dialect/CommonIR/Transforms/BufferLayoutPropagation.h"
#include "mlir-ext/Dialect/CommonIR/Transforms/TensorLayoutPropagation.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::metax {

// Plan shared storage from existing dot operand layouts without changing IR.
FailureOr<tile::BufferLayoutPlan>
planSharedLayouts(ModuleOp module, ArrayRef<triton::DotOp> dots,
                  const tile::TensorLayoutRequirements &writerPreferences);

} // namespace mlir::triton::metax
#endif
