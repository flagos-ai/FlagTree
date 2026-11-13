#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIROPTIMIZETENSORTILELAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

struct CommonIROptimizeTensorTileLayoutsPass
    : impl::CommonIROptimizeTensorTileLayoutsBase<
          CommonIROptimizeTensorTileLayoutsPass> {
  void runOnOperation() override {
    TensorLayoutRequirements requirements;
    getOperation().walk([&](gpu::ConvertLayoutOp convert) {
      if (tile::getExplicitResultEncoding(convert, 0))
        return;
      // Leave ordinary conversions to RemoveLayoutConversions.
      llvm::SetVector<Operation *> slice;
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      options.filter = [](Operation *op) {
        return op->getNumRegions() == 0 && isMemoryEffectFree(op);
      };
      if (failed(getBackwardSlice(convert.getOperation(), &slice, options)) ||
          !llvm::any_of(slice, [](Operation *op) {
            return isa<tile::ExtractTileOp, tile::InsertTileOp>(op);
          }))
        return;
      Attribute encoding = convert.getType().getEncoding();
      for (OpOperand &use : convert.getResult().getUses()) {
        // Intermediate targets can unnecessarily fix loop layouts.
        // Propagate only the final consumer's requirement.
        if (isa<gpu::ConvertLayoutOp>(use.getOwner()))
          continue;
        requirements.uses.push_back({&use, encoding});
      }
    });
    if (requirements.uses.empty())
      return;
    TTGIRLayoutAdapter adapter;
    applyLayoutRequirements(getOperation(), requirements, adapter);
  }
};

} // namespace
} // namespace mlir::triton::metax
