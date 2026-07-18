// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//===- LiftTTCFToSCF.cpp ---------------------------------------*- C++ -*-===//
//
// Mostly inherited from mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.cpp
// reason is cfToSCF only supports func.funcOp, we need to operate on tt.funcOp
// Apply MLIR ControlFlowToSCF transformation inside Triton tt.func.
//
//===----------------------------------------------------------------------===//

#include "Transform/Passes.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CFGToSCF.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_LIFTTTCFTOSCF
#include "Transform/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// A ControlFlowToSCF transformation that creates tt.return for unreachable.
struct TTControlFlowToSCFTransformation
    : public ControlFlowToSCFTransformation {
  FailureOr<Operation *> createUnreachableTerminator(Location loc,
                                                     OpBuilder &builder,
                                                     Region &region) override {
    Operation *parentOp = region.getParentOp();
    if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
      SmallVector<Value> rets;
      for (Type ty : funcOp.getResultTypes())
        rets.push_back(getUndefValue(loc, builder, ty));
      return triton::ReturnOp::create(builder, loc, rets).getOperation();
    }
    return ControlFlowToSCFTransformation::createUnreachableTerminator(
        loc, builder, region);
  }
};

struct LiftTTCFToSCFPass
    : public ::mlir::triton::impl::LiftTTCFToSCFBase<LiftTTCFToSCFPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    TTControlFlowToSCFTransformation transformation;
    bool changed = false;

    WalkResult walkRes = module.walk([&](triton::FuncOp funcOp) {
      if (funcOp.getBody().empty())
        return WalkResult::advance();

      auto &domInfo = funcOp != module ? getChildAnalysis<DominanceInfo>(funcOp)
                                       : getAnalysis<DominanceInfo>();

      auto visitor = [&](Operation *innerOp) -> WalkResult {
        for (Region &reg : innerOp->getRegions()) {
          FailureOr<bool> changedFunc =
              transformCFGToSCF(reg, transformation, domInfo);
          if (failed(changedFunc))
            return WalkResult::interrupt();
          changed |= *changedFunc;
        }
        return WalkResult::advance();
      };

      if (funcOp->walk<WalkOrder::PostOrder>(visitor).wasInterrupted())
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return signalPassFailure();
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

namespace mlir::triton {
std::unique_ptr<Pass> createLiftTTCFToSCFPass() {
  return std::make_unique<LiftTTCFToSCFPass>();
}
} // namespace mlir::triton
