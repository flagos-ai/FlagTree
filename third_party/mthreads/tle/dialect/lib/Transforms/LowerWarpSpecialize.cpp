#ifdef __TLE__

#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEPREPAREWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
class PrepareWarpSpecializePass
    : public impl::TritonMUSAGPUTLEPrepareWarpSpecializeBase<
          PrepareWarpSpecializePass> {
  LogicalResult prepareWarpSpecialize(ModuleOp mod, ttg::WarpSpecializeOp ws,
                                      IRRewriter &rewriter) {
    auto func = ws->getParentOfType<triton::FuncOp>();
    if (!func)
      return ws.emitOpError("requires a parent Triton function");

    Block &entry = func.getBody().front();
    if (ws->getBlock() != &entry)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize must be in the function entry "
          "block");
    if (!ws->getNextNode() || !isa<triton::ReturnOp>(ws->getNextNode()) ||
        ws->getNextNode()->getNextNode())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize must be the final operation "
          "before tt.return");
    if (ws.getNumResults() != 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize does not support results");

    auto partitions = ws.getPartitionOp();
    if (partitions.getPartitionRegions().size() != 1)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize requires exactly one producer "
          "partition");
    if (ws.getPartitionNumWarps().size() != 1 ||
        ws.getPartitionNumWarps().front() <= 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize requires a positive static "
          "producer warp count");

    Region &consumerRegion = ws.getDefaultRegion();
    Region &producerRegion = partitions.getPartitionRegions().front();
    if (!consumerRegion.hasOneBlock() || !producerRegion.hasOneBlock())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize requires single-block "
          "consumer and producer regions");

    Block &consumerBlock = consumerRegion.front();
    Block &producerBlock = producerRegion.front();
    auto consumerYield =
        dyn_cast<ttg::WarpYieldOp>(consumerBlock.getTerminator());
    if (!consumerYield || consumerYield.getNumOperands() != 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize consumer must yield no values");
    if (!isa<ttg::WarpReturnOp>(producerBlock.getTerminator()))
      return ws.emitOpError(
          "mthreads TLE static warp_specialize producer must end with "
          "ttg.warp_return");

    ValueRange captures = partitions.getExplicitCaptures();
    if (producerBlock.getNumArguments() != captures.size())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize producer capture count "
          "mismatch");
    DominanceInfo dominance(func);
    for (Value capture : captures) {
      if (!dominance.dominates(capture, ws.getOperation()))
        return ws.emitOpError(
            "mthreads TLE static warp_specialize capture must dominate the "
            "partition split");
    }

    int32_t baseNumWarps = ttg::lookupNumWarps(ws);
    if (baseNumWarps <= 0)
      return ws.emitOpError(
          "mthreads TLE static warp_specialize consumer warp count must be "
          "positive");
    int32_t producerWarps = ws.getPartitionNumWarps().front();
    int32_t warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    int64_t totalNumWarps64 =
        static_cast<int64_t>(baseNumWarps) + producerWarps;
    int64_t producerBegin64 = static_cast<int64_t>(baseNumWarps) * warpSize;
    int64_t totalThreads64 = totalNumWarps64 * warpSize;
    if (warpSize <= 0 ||
        totalNumWarps64 > std::numeric_limits<int32_t>::max() ||
        producerBegin64 > std::numeric_limits<int32_t>::max() ||
        totalThreads64 > std::numeric_limits<int32_t>::max())
      return ws.emitOpError(
          "mthreads TLE static warp_specialize thread count overflow");

    int32_t totalNumWarps = static_cast<int32_t>(totalNumWarps64);
    if (auto existingStartIds = ws.getWarpGroupStartIds()) {
      if (existingStartIds->size() != 1 ||
          existingStartIds->front() != baseNumWarps)
        return ws.emitOpError(
            "mthreads TLE static producer must begin after the default "
            "partition");
    } else {
      ws.setWarpGroupStartIds({baseNumWarps});
    }
    if (auto existing =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
      if (existing.getInt() != totalNumWarps)
        return ws.emitOpError(
            "mthreads TLE static warp_specialize conflicts with existing "
            "ttg.total-num-warps");
    } else {
      mod->setAttr("ttg.total-num-warps",
                   rewriter.getI32IntegerAttr(totalNumWarps));
    }

    return success();
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty())
      return;
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "mthreads TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(prepareWarpSpecialize(mod, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
