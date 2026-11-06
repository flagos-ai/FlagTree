#include "triton/Conversion/CommonIRToTTGIR/SharedLayoutUtils.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIROPTIMIZESHAREDLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

struct CommonIROptimizeSharedLayoutsPass
    : impl::CommonIROptimizeSharedLayoutsBase<CommonIROptimizeSharedLayoutsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::DotOp> dots;
    module.walk([&](triton::DotOp dot) {
      auto mma = dyn_cast_or_null<gpu::MACAMmaEncodingAttr>(
          dot.getType().getEncoding());
      auto elementType = dot.getA().getType().getElementType();
      if (!mma || mma.getVersionMajor() != 2 || mma.getIsATrans() ||
          mma.getIsBTrans() || (!elementType.isF16() && !elementType.isBF16()))
        return;
      dots.push_back(dot);
    });
    if (dots.empty())
      return;

    // Writer preferences guide shared order even when logical tiles hide
    // global contiguity from the ordinary Coalesce pass.
    TensorLayoutRequirements writerPreferences;
    collectCopyWriterPreferences(module, writerPreferences);
    collectLoadStoreLayoutRequirements(module, writerPreferences);
    auto buffers = planSharedLayouts(module, dots, writerPreferences);
    if (failed(buffers))
      return signalPassFailure();

    // Keep register encodings fixed here. CommonIRPropagateLayouts reconciles
    // the copy and ordinary load/store requirements in the next step.
    buffers->apply();
  }
};

} // namespace
} // namespace mlir::triton::metax
