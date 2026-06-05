/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>

using namespace mlir;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUOptimizeCStorePass
    : public TritonMETAXGPUOptimizeCStorePassBase<
          TritonMETAXGPUOptimizeCStorePass> {
public:
  TritonMETAXGPUOptimizeCStorePass() = default;
  TritonMETAXGPUOptimizeCStorePass(int numStages) {
    this->numStages = numStages;
  }
  void runOnOperation() override {
    // When numStages == 1, eliminate the convetLayout for now.
    ModuleOp m = getOperation();
    if (this->numStages != 1) {
      return;
    }
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      OpBuilder builder(forOp);

      for (unsigned i = 0; i < forOp->getNumResults(); ++i) {
        scf::YieldOp yieldOp =
            cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (forOp->getResult(i).hasOneUse() &&
            isa<tt::DotOp>(yieldOp->getOperand(i).getDefiningOp())) {
          Value res = forOp->getResult(i);
          Operation *Op = *res.getUsers().begin();
          while (!isa<tt::StoreOp>(Op) && Op->getResult(0).hasOneUse()) {
            Op = *(Op->getUsers().begin());
          }
          // TODO: deal with if-elseOp between forOp and storeOp
          // For example:
          // for loop:
          //   C = tl.dot(A, B, C)
          // if flag:
          //   C *= scale
          // tl.store(c_ptr, C)
          // (and flag is not tl.constexpr)
          if (tt::StoreOp storeOp = dyn_cast<tt::StoreOp>(Op)) {
            OptimizeCStore(builder, storeOp);
          }
        } else
          continue;
      }
      return;
    });
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUOptimizeCStorePass(int numStages) {
  return std::make_unique<TritonMETAXGPUOptimizeCStorePass>(numStages);
}
