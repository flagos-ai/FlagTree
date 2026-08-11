#ifndef TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_TRITON_PTR_TO_MEMREF_H
#define TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_TRITON_PTR_TO_MEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonPtrToMemrefPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_TRITON_PTR_TO_MEMREF_H
