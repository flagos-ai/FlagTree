#ifndef TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
#define TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_TRITONTOSTRUCTURED
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToStructuredPass(bool enableMakeGatherScatterTensorPtr = true);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
