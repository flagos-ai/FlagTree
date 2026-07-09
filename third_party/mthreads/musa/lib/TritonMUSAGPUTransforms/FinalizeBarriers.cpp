#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUFINALIZEBARRIERS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUFinalizeBarriersPass
    : impl::TritonMUSAGPUFinalizeBarriersBase<
          TritonMUSAGPUFinalizeBarriersPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    for (tt::FuncOp func : mod.getOps<tt::FuncOp>())
      triton::musa::finalizeBarRecord(func, rewriter);
  }
};

} // namespace mlir
