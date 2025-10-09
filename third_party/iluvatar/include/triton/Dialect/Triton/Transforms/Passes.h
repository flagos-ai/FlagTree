#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

#include "flagtree_spec.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createCombineOpsPass();

std::unique_ptr<Pass> createReorderBroadcastPass();
std::unique_ptr<Pass> createRewriteTensorPointerPass();

#ifndef FLAGTREE_SPEC_Dialect_Triton_Transforms_Passes_createExpressionRestructingPass
std::unique_ptr<Pass> createExpressionRestructingPass();
#endif
} // namespace triton

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir

#endif
