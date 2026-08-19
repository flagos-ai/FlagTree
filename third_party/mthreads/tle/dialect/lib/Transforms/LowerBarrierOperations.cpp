#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERBARRIEROPERATIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttmg = triton::musa;
namespace musa_tle = triton::musa_tle;

class LowerBarrierOperationsPass
    : public impl::TritonMUSAGPUTLELowerBarrierOperationsBase<
          LowerBarrierOperationsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<musa_tle::BarrierWaitOp> waits;
    SmallVector<musa_tle::BarrierArriveOp> arrivals;
    module.walk([&](musa_tle::BarrierWaitOp op) { waits.push_back(op); });
    module.walk([&](musa_tle::BarrierArriveOp op) { arrivals.push_back(op); });

    for (musa_tle::BarrierWaitOp wait : waits) {
      rewriter.setInsertionPoint(wait);
      auto lowered = ttmg::WaitBarrierOp::create(
          rewriter, wait.getLoc(), wait.getBarId(), wait.getPhase());
      lowered->setDiscardableAttrs(wait->getDiscardableAttrDictionary());
      rewriter.eraseOp(wait);
    }

    for (musa_tle::BarrierArriveOp arrive : arrivals) {
      if (arrive.getArriveCount() != 1) {
        arrive.emitOpError(
            "mthreads hardware barrier arrive requires arrive_count = 1");
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(arrive);
      auto lowered = ttmg::WarpArriveBarrierOp::create(
          rewriter, arrive.getLoc(), arrive.getBarId(), arrive.getPhase());
      lowered->setDiscardableAttrs(arrive->getDiscardableAttrDictionary());
      rewriter.eraseOp(arrive);
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
