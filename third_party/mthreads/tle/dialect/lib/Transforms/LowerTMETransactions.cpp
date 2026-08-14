#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERTMETRANSACTIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static FailureOr<int32_t>
resolveIssueThread(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  auto ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
  if (!ws)
    return 0;

  Region *copyRegion = copy->getParentRegion();
  if (ws.getDefaultRegion().isAncestor(copyRegion)) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the producer partition");
    return failure();
  }

  std::optional<unsigned> partitionIndex;
  for (auto [index, region] : llvm::enumerate(ws.getPartitionRegions())) {
    if (region->isAncestor(copyRegion)) {
      partitionIndex = index;
      break;
    }
  }
  if (!partitionIndex || *partitionIndex != 0) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the producer partition");
    return failure();
  }

  ModuleOp module = copy->getParentOfType<ModuleOp>();
  auto numWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarp =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
      threadsPerWarp.getInt() <= 0) {
    copy.emitOpError("mthreads TLE producer issue thread requires "
                     "ttg.num-warps and ttg.threads-per-warp");
    return failure();
  }

  int64_t issueThread = numWarps.getInt() * threadsPerWarp.getInt();
  if (issueThread > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("mthreads TLE producer issue thread exceeds int32 range");
    return failure();
  }
  return static_cast<int32_t>(issueThread);
}

class LowerTMETransactionsPass
    : public impl::TritonMUSAGPUTLELowerTMETransactionsBase<
          LowerTMETransactionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> copies;
    module.walk([&](ttmg::AsyncTMECopyGlobalToLocalOp copy) {
      if (copy->hasAttr(ttmg::kTLEExpectBytesAttr))
        copies.push_back(copy);
    });

    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : copies) {
      auto expectBytes =
          copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
      if (!expectBytes || !expectBytes.getType().isInteger(32) ||
          expectBytes.getInt() <= 0) {
        copy.emitOpError(
            "mthreads TLE completion TME copy requires positive expect_bytes");
        signalPassFailure();
        return;
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(copy);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }

      Location loc = copy.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();

      rewriter.setInsertionPoint(copy);
      Value bytes =
          arith::ConstantIntOp::create(rewriter, loc, expectBytes.getInt(), 32);
      auto addTrans = ttmg::BarrierAddTransOp::create(
          rewriter, loc, copy.getBarId(), bytes, copy.getPred());

      rewriter.setInsertionPointAfter(copy);
      auto arrive = ttmg::ArriveBarrierNoRetOp::create(
          rewriter, loc, copy.getBarId(), copy.getPred());

      for (Operation *op : {addTrans.getOperation(), copy.getOperation(),
                            arrive.getOperation()}) {
        op->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        op->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
      }
      copy->removeAttr(ttmg::kTLEExpectBytesAttr);
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
