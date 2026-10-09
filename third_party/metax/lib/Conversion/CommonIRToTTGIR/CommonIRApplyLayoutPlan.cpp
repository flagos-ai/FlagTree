// MIT License
// Copyright (c) 2026 The FlagOS Contributors
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIRAPPLYLAYOUTPLAN
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {
void collectDotLayoutRequirements(
    ArrayRef<DotLayoutChoice> choices,
    tile::TensorLayoutRequirements &requirements) {
  for (auto [dot, plan] : choices) {
    requirements.assignments[dot.getResult()] = plan.mma;
    // Hardware port requirements alone do not rematerialize their inputs.
    // Producer rules and C propagation are added separately below.
    SmallVector<Attribute> encodings{plan.operandA, plan.operandB};
    for (auto [index, encoding] : llvm::enumerate(encodings))
      requirements.uses.push_back({&dot->getOpOperand(index), encoding, false});
  }
}

// Only the hardware legality of logical tiles is target-specific. C uses the
// same producer rematerialization, loop planning and boundaries as copy.
struct MetaxLayoutAdapter final : tile::TTGIRLayoutAdapter {
  Attribute inferOperandEncoding(Operation *op,
                                 Attribute encoding) const override {
    if (isa<tile::ExtractTileOp, tile::InsertTileOp>(op) &&
        isa<gpu::BlockedEncodingAttr, gpu::MACAMmaEncodingAttr>(encoding))
      return canLowerTensorTileInEncoding(op, encoding) ? encoding
                                                        : Attribute{};
    return TTGIRLayoutAdapter::inferOperandEncoding(op, encoding);
  }
};

struct CommonIRApplyLayoutPlanPass
    : impl::CommonIRApplyLayoutPlanBase<CommonIRApplyLayoutPlanPass> {
  using impl::CommonIRApplyLayoutPlanBase<
      CommonIRApplyLayoutPlanPass>::CommonIRApplyLayoutPlanBase;
  CommonIRApplyLayoutPlanPass() = default;
  CommonIRApplyLayoutPlanPass(CommonIRApplyLayoutPlanOptions options,
                              llvm::StringMap<DotLayoutPlan> plans)
      : CommonIRApplyLayoutPlanBase(options), selectedPlans(std::move(plans)) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<DotLayoutChoice> choices;
    if (selectedPlans) {
      auto selected =
          selectDotLayouts(module, computeCapability, numWarps, *selectedPlans);
      if (failed(selected))
        return signalPassFailure();
      choices = std::move(*selected);
    } else {
      // The default compiler keeps its existing MMA selector. Before that
      // selector runs only memory requirements are available to this pass.
      module.walk([&](triton::DotOp dot) {
        auto mma = dot.getType().getEncoding();
        if (isa_and_nonnull<gpu::MACAMmaEncodingAttr>(mma))
          choices.push_back({dot,
                             {mma, dot.getA().getType().getEncoding(),
                              dot.getB().getType().getEncoding()}});
      });
    }

    tile::TensorLayoutRequirements requirements;
    collectDotLayoutRequirements(choices, requirements);
    tile::collectCopyLayoutRequirements(module, requirements);
    tile::collectLoadStoreLayoutRequirements(module, requirements);
    auto buffers = planSharedLayouts(module, choices, requirements);
    if (failed(buffers))
      return signalPassFailure();
    if ((selectedPlans || !choices.empty()) &&
        failed(collectBsmLayoutRequirements(module, choices, requirements)))
      return signalPassFailure();
    collectLocalLoadLayoutRequirements(choices, requirements, *buffers);
    // The dot fixes C's layout; propagate it through legal logical tiles and
    // loop carriers. Incompatible producers remain explicit conversions.
    for (auto [dot, plan] : choices)
      requirements.uses.push_back({&dot->getOpOperand(2), plan.mma});

    // Collection and propagation are read-only. No pass is run recursively;
    // one shared plan owns the producer, loop and conversion rewrites.
    if (!requirements.uses.empty() || !requirements.assignments.empty()) {
      MetaxLayoutAdapter adapter;
      buffers->apply();
      tile::applyLayoutRequirements(module, requirements, adapter);
    }
  }

private:
  std::optional<llvm::StringMap<DotLayoutPlan>> selectedPlans;
};
} // namespace

std::unique_ptr<Pass>
createCommonIRApplyLayoutPlan(CommonIRApplyLayoutPlanOptions options,
                              llvm::StringMap<DotLayoutPlan> selectedPlans) {
  return std::make_unique<CommonIRApplyLayoutPlanPass>(
      options, std::move(selectedPlans));
}
} // namespace mlir::triton::metax
