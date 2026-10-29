#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIRPROPAGATELAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

struct CommonIRPropagateLayoutsPass
    : impl::CommonIRPropagateLayoutsBase<CommonIRPropagateLayoutsPass> {
  void runOnOperation() override {
    TensorLayoutRequirements requirements;
    collectLegalCopyLayoutRequirements(getOperation(), requirements);
    collectLoadStoreLayoutRequirements(getOperation(), requirements);
    if (requirements.uses.empty())
      return;
    // Reconcile memory requirements while tiles still describe logical views.
    // Shared layouts and explicit tensor anchors are preserved by the adapters.
    TTGIRLayoutAdapter adapter;
    applyLayoutRequirements(getOperation(), requirements, adapter);
  }
};

} // namespace
} // namespace mlir::triton::metax
