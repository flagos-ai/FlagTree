//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonxpu-tle-legalize"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTLELEGALIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUTLELegalizePass
    : public impl::TritonXPUTLELegalizeBase<TritonXPUTLELegalizePass> {

  using impl::TritonXPUTLELegalizeBase<
      TritonXPUTLELegalizePass>::TritonXPUTLELegalizeBase;

  TritonXPUTLELegalizePass() = default;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    // --- TLE reduce lowering ---------------------------------------------
    // The TLE pipeline does not run the normal tritonxpu-legalize pass, so
    // tt.reduce is never turned into triton_xpu.reduce (that transform lives in
    // Legalize.cpp and is entangled with core-tiling loops). For the current
    // TLE model (fixed [1, BLOCK] tile, no core-tiling), a whole tile lives in
    // one iteration, so we can convert tt.reduce -> triton_xpu.reduce directly
    // with loopNum=1 and loopIndex=0.
    SmallVector<triton::ReduceOp> ttReduceOps;
    m.walk([&](triton::ReduceOp op) { ttReduceOps.push_back(op); });
    for (auto reduceOp : ttReduceOps) {
      OpBuilder builder(reduceOp);
      auto loc = reduceOp->getLoc();
      Value loopIndex = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
      auto newReduceOp = builder.create<triton::xpu::ReduceOp>(
          loc, reduceOp->getResultTypes(), reduceOp.getSrcs(),
          reduceOp.getAxis(), /*loopNum=*/1, loopIndex);
      auto &newCombineOp = newReduceOp.getCombineOp();
      builder.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp,
                                newCombineOp.end());
      // tt.reduce.return -> triton_xpu.reduce.return inside the combine region.
      for (auto &opInCombine :
           llvm::make_early_inc_range(newCombineOp.getOps())) {
        if (auto redReturnOp =
                dyn_cast<mlir::triton::ReduceReturnOp>(&opInCombine)) {
          OpBuilder retBuilder(redReturnOp);
          auto newRedReturnOp = retBuilder.create<triton::xpu::ReduceReturnOp>(
              redReturnOp.getLoc(), redReturnOp.getOperands());
          redReturnOp->replaceAllUsesWith(newRedReturnOp->getResults());
          redReturnOp.erase();
        }
      }
      reduceOp->replaceAllUsesWith(newReduceOp->getResults());
      reduceOp->erase();
    }

    // Count reduce/scan ops for helper id assignment below. (TLE elementwise
    // kernels have none; kept so ReduceOpHelper/ScanLoweringHelper stay valid.)
    unsigned reduceId = 0;
    unsigned reduceNum = 0;
    unsigned scanId = 0;
    unsigned scanNum = 0;
    m.walk([&](triton::xpu::ReduceOp) { reduceNum++; });
    m.walk([&](triton::xpu::ScanOp) { scanNum++; });

    // Set ReduceOpHelper
    m.walk([&](triton::xpu::ReduceOp redOp) {
      ReduceOpHelper helper(redOp);
      helper.setReduceId(reduceId);
      helper.setReduceNum(reduceNum);
      reduceId++;
    });

    // Set ScanLoweringHelper
    m.walk([&](triton::xpu::ScanOp scanOp) {
      ScanLoweringHelper helper(scanOp);
      helper.setScanId(scanId);
      helper.setScanNum(scanNum);
      scanId++;
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
