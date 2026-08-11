#ifndef TRITON_CONVERSION_TRITON_TO_UNSTRUCTURED_TRITON_TO_UNSTRUCTURED_H
#define TRITON_CONVERSION_TRITON_TO_UNSTRUCTURED_TRITON_TO_UNSTRUCTURED_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToUnstructuredPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_UNSTRUCTURED_TRITON_TO_UNSTRUCTURED_H
