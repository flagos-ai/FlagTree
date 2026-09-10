#ifdef __TLE__

#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
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
          "MUSA TLE static warp_specialize must be in the function entry "
          "block");
    if (!ws->getNextNode() || !isa<triton::ReturnOp>(ws->getNextNode()) ||
        ws->getNextNode()->getNextNode())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize must be the final operation "
          "before tt.return");
    if (ws.getNumResults() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize does not support results");

    auto partitions = ws.getPartitionOp();
    auto workerRegions = partitions.getPartitionRegions();
    ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
    if (workerRegions.empty())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires at least one worker "
          "partition");
    if (workerRegions.size() != workerWarps.size())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize worker region and warp-count "
          "sizes must match");
    if (auto requestedRegisters = ws.getRequestedRegisters()) {
      if (requestedRegisters->size() != workerRegions.size())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize requested-register count must "
            "match the worker partition count");
    }

    Region &defaultRegion = ws.getDefaultRegion();
    if (!defaultRegion.hasOneBlock())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires a single-block default "
          "region");

    Block &defaultBlock = defaultRegion.front();
    auto defaultYield =
        dyn_cast<ttg::WarpYieldOp>(defaultBlock.getTerminator());
    if (!defaultYield || defaultYield.getNumOperands() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default region must yield no "
          "values");

    ValueRange captures = partitions.getExplicitCaptures();
    for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
      if (!workerRegion.hasOneBlock())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must be single-block";
      Block &workerBlock = workerRegion.front();
      if (!isa<ttg::WarpReturnOp>(workerBlock.getTerminator()))
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must end with ttg.warp_return";
      if (workerBlock.getNumArguments() != captures.size())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " capture count mismatch";
      for (auto [captureIndex, argumentAndCapture] :
           llvm::enumerate(llvm::zip(workerBlock.getArguments(), captures))) {
        auto [argument, capture] = argumentAndCapture;
        if (argument.getType() != capture.getType())
          return ws.emitOpError()
                 << "MUSA TLE static warp_specialize worker partition #"
                 << workerIndex << " capture #" << captureIndex
                 << " type mismatch";
      }
    }
    DominanceInfo dominance(func);
    for (Value capture : captures) {
      if (!dominance.dominates(capture, ws.getOperation()))
        return ws.emitOpError(
            "MUSA TLE static warp_specialize capture must dominate the "
            "partition split");
    }

    int32_t baseNumWarps = ttg::lookupNumWarps(ws);
    if (baseNumWarps <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default warp count must be "
          "positive");
    int32_t warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    if (warpSize <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize warp size must be positive");

    int64_t nextWarp = baseNumWarps;
    SmallVector<int32_t> workerStartIds;
    workerStartIds.reserve(workerWarps.size());
    for (auto [workerIndex, numWarps] : llvm::enumerate(workerWarps)) {
      if (numWarps <= 0)
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " requires a positive static warp count";
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
      workerStartIds.push_back(static_cast<int32_t>(nextWarp));
      nextWarp += numWarps;
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
    }
    if (nextWarp > std::numeric_limits<int32_t>::max() / warpSize)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize thread count exceeds int32");

    int32_t totalNumWarps = static_cast<int32_t>(nextWarp);
    if (auto existingStartIds = ws.getWarpGroupStartIds()) {
      if (existingStartIds->size() != workerStartIds.size() ||
          !llvm::equal(*existingStartIds, workerStartIds))
        return ws.emitOpError(
            "MUSA TLE static worker warp ranges must follow declaration "
            "order after the default partition");
    } else {
      ws.setWarpGroupStartIds(workerStartIds);
    }
    if (auto existing =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
      if (existing.getInt() != totalNumWarps)
        return ws.emitOpError(
            "MUSA TLE static warp_specialize conflicts with existing "
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
          "MUSA TLE static warp_specialize supports exactly one marked "
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
