#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIRPLANSHAREDLAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

struct CommonIRPlanSharedLayoutsPass
    : impl::CommonIRPlanSharedLayoutsBase<CommonIRPlanSharedLayoutsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<DotLayoutChoice> choices;
    module.walk([&](triton::DotOp dot) {
      auto mma = dyn_cast_or_null<gpu::MACAMmaEncodingAttr>(
          dot.getType().getEncoding());
      auto elementType = dot.getA().getType().getElementType();
      if (!mma || mma.getVersionMajor() != 2 || mma.getIsATrans() ||
          mma.getIsBTrans() || (!elementType.isF16() && !elementType.isBF16()))
        return;
      choices.push_back({dot,
                         {mma, dot.getA().getType().getEncoding(),
                          dot.getB().getType().getEncoding()}});
    });
    if (choices.empty())
      return;

    // Writer preferences guide shared order even when logical tiles hide
    // global contiguity from the ordinary Coalesce pass.
    TensorLayoutRequirements writerPreferences;
    collectCopyWriterPreferences(module, writerPreferences);
    collectLoadStoreLayoutRequirements(module, writerPreferences);
    auto buffers = planSharedLayouts(module, choices, writerPreferences);
    if (failed(buffers))
      return signalPassFailure();

    // Keep register encodings fixed here. CommonIRPropagateLayouts reconciles
    // the copy and ordinary load/store requirements in the next step.
    buffers->apply();
  }
};

} // namespace
} // namespace mlir::triton::metax
