// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/Passes.h"

#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::triton::metax {

#define GEN_PASS_DEF_COMMONIRINJECTDOTPLAN
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

// Rewrite a single tt.dot to use the injected {mma, operandA, operandB} plan,
// mirroring the structural change that accelerate-matmul performs but sourcing
// the MMA encoding from our own dot plan instead of the target's auto choice.
static void injectPlan(tt::DotOp dot, const DotLayoutPlan &plan) {
  OpBuilder builder(dot);
  Location loc = dot.getLoc();

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());

  auto mmaTy =
      RankedTensorType::get(dTy.getShape(), dTy.getElementType(), plan.mma);
  auto newAType = RankedTensorType::get(aTy.getShape(), aTy.getElementType(),
                                        plan.operandA);
  auto newBType = RankedTensorType::get(bTy.getShape(), bTy.getElementType(),
                                        plan.operandB);

  Value newA = builder.create<ttg::ConvertLayoutOp>(loc, newAType, dot.getA());
  Value newB = builder.create<ttg::ConvertLayoutOp>(loc, newBType, dot.getB());
  Value newC = builder.create<ttg::ConvertLayoutOp>(loc, mmaTy, dot.getC());

  // Preserve precision, accumulation limits and other dot attributes.
  auto newDot = cast<tt::DotOp>(builder.clone(*dot.getOperation()));
  newDot->setOperands({newA, newB, newC});
  newDot.getResult().setType(mmaTy);

  auto backCvt =
      builder.create<ttg::ConvertLayoutOp>(loc, dTy, newDot.getResult());
  dot.getResult().replaceAllUsesWith(backCvt.getResult());
  dot.erase();
}

struct CommonIRInjectDotPlanPass
    : public impl::CommonIRInjectDotPlanBase<CommonIRInjectDotPlanPass> {
  using impl::CommonIRInjectDotPlanBase<
      CommonIRInjectDotPlanPass>::CommonIRInjectDotPlanBase;

  CommonIRInjectDotPlanPass() = default;
  CommonIRInjectDotPlanPass(CommonIRInjectDotPlanOptions options,
                            llvm::StringMap<DotLayoutPlan> plans)
      : CommonIRInjectDotPlanBase(options), selectedPlans(std::move(plans)) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto moduleWarps =
        module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
    if (!moduleWarps || moduleWarps.getInt() != numWarps) {
      module.emitError("dot plan injection requires prepared TTGIR with the "
                       "requested warp count");
      return signalPassFailure();
    }
    auto domains = enumerateDotLayoutPlans(module, computeCapability, numWarps);
    if (failed(domains))
      return signalPassFailure();
    if (selectedPlans && selectedPlans->size() != domains->size()) {
      module.emitError("candidate must select exactly one plan for every dot");
      return signalPassFailure();
    }

    // Validate the complete assignment before changing any dot.
    SmallVector<DotLayoutPlan> choices;
    for (const auto &domain : *domains) {
      auto dot = domain.dot;
      for (Value value : {dot.getA(), dot.getB(), dot.getC(), dot.getD()}) {
        if (!cast<RankedTensorType>(value.getType()).getEncoding()) {
          dot.emitError("dot plan injection requires encoded TTGIR tensors");
          return signalPassFailure();
        }
      }
      DotLayoutPlan choice = domain.plans.front();
      if (selectedPlans) {
        std::string dotId =
            domain.functionName + "/dot/" + std::to_string(domain.dotIndex);
        auto it = selectedPlans->find(dotId);
        if (it == selectedPlans->end()) {
          dot.emitError("candidate is missing a plan for ") << dotId;
          return signalPassFailure();
        }
        choice = it->second;
        if (!llvm::any_of(domain.plans, [&](const DotLayoutPlan &legal) {
              return choice.mma == legal.mma &&
                     choice.operandA == legal.operandA &&
                     choice.operandB == legal.operandB;
            })) {
          dot.emitError("selected layout is not a legal enumerated plan for ")
              << dotId;
          return signalPassFailure();
        }
      }
      if (isa<ttg::MACAMmaEncodingAttr>(
              cast<RankedTensorType>(dot.getD().getType()).getEncoding())) {
        dot.emitError("dot layout plan has already been injected");
        return signalPassFailure();
      }
      choices.push_back(choice);
    }
    for (auto [domain, choice] : llvm::zip(*domains, choices))
      injectPlan(domain.dot, choice);
  }

private:
  std::optional<llvm::StringMap<DotLayoutPlan>> selectedPlans;
};

} // namespace

std::unique_ptr<Pass>
createCommonIRInjectDotPlan(CommonIRInjectDotPlanOptions options,
                            llvm::StringMap<DotLayoutPlan> selectedPlans) {
  return std::make_unique<CommonIRInjectDotPlanPass>(options,
                                                     std::move(selectedPlans));
}

} // namespace mlir::triton::metax
