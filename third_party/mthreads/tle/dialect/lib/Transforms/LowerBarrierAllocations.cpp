#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERBARRIERALLOCATIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

using triton::musa_tle::BarrierAllocOp;
using triton::musa_tle::BarrierIndexOp;
namespace ttg = triton::gpu;

static constexpr StringLiteral kExhaustedDiagnostic =
    "mthreads TLE barrier allocation exhausted hardware barrier ids";

class LowerBarrierAllocationsPass
    : public impl::TritonMUSAGPUTLELowerBarrierAllocationsBase<
          LowerBarrierAllocationsPass> {
  LogicalResult lowerFunction(triton::FuncOp func, IRRewriter &rewriter) {
    SmallVector<BarrierAllocOp> allocs;
    func.walk([&](BarrierAllocOp op) { allocs.push_back(op); });
    if (allocs.empty())
      return success();

    Block &entry = func.getBody().front();
    SmallVector<BarrierAllocOp> entryAllocs;
    for (Operation &op : entry) {
      if (auto alloc = dyn_cast<BarrierAllocOp>(op))
        entryAllocs.push_back(alloc);
    }
    if (entryAllocs.size() != allocs.size()) {
      auto nested = llvm::find_if(allocs, [&](BarrierAllocOp alloc) {
        return alloc->getBlock() != &entry;
      });
      return nested->emitOpError(
          "mthreads TLE barrier initialization requires allocations in the "
          "function entry block");
    }
    allocs = std::move(entryAllocs);

    ttg::WarpSpecializeOp warpSpecialize;
    bool hasWarpSpecialize = false;
    func.walk([&](ttg::WarpSpecializeOp) { hasWarpSpecialize = true; });
    for (Operation &op : entry) {
      if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        warpSpecialize = ws;
        break;
      }
    }

    if (hasWarpSpecialize) {
      if (!warpSpecialize) {
        return allocs.front().emitOpError(
            "mthreads TLE barrier initialization for warp_specialize requires "
            "a top-level warp_specialize in the function entry block");
      }
      for (BarrierAllocOp alloc : allocs) {
        if (alloc->getBlock() != &entry ||
            !alloc->isBeforeInBlock(warpSpecialize)) {
          return alloc.emitOpError(
              "mthreads TLE barrier initialization for warp_specialize "
              "requires barrier allocations in the function entry block "
              "before the first top-level warp_specialize");
        }
      }
    }

    BarrierAllocOp lastAlloc = allocs.back();
    DominanceInfo dominance(func);
    for (BarrierAllocOp alloc : allocs) {
      for (Operation *user : alloc.getBaseId().getUsers()) {
        if (!dominance.dominates(lastAlloc.getOperation(), user)) {
          return alloc.emitOpError(
              "mthreads TLE barrier initialization requires all allocations "
              "before the first barrier use");
        }
      }
    }

    int64_t totalBarriers = 0;
    for (BarrierAllocOp alloc : allocs) {
      totalBarriers += alloc.getNumBarriers();
      if (totalBarriers > std::numeric_limits<int32_t>::max())
        break;
    }

    if (totalBarriers <= 0 ||
        totalBarriers > std::numeric_limits<int32_t>::max()) {
      allocs.front().emitOpError()
          << kExhaustedDiagnostic << ": cannot reserve " << totalBarriers
          << " additional ids in [1, " << triton::musa::kMaxBarrierId << "]";
      return failure();
    }

    auto reserved = triton::musa::reserveBarrierIdRange(
        allocs.front(), static_cast<int32_t>(totalBarriers));
    if (failed(reserved)) {
      allocs.front().emitOpError()
          << kExhaustedDiagnostic << ": cannot reserve " << totalBarriers
          << " additional ids in [1, " << triton::musa::kMaxBarrierId << "]";
      return failure();
    }

    SmallVector<BarrierIndexOp> indices;
    func.walk([&](BarrierIndexOp op) { indices.push_back(op); });

    Operation *lastInitArrival = nullptr;
    int32_t nextBase = *reserved;
    for (BarrierAllocOp alloc : allocs) {
      Location loc = alloc.getLoc();
      rewriter.setInsertionPoint(alloc);
      Value base = arith::ConstantIntOp::create(rewriter, loc, nextBase, 32);
      Value arriveCount = arith::ConstantIntOp::create(
          rewriter, loc, alloc.getArriveCount(), 32);
      Value initPolarity = arith::ConstantIntOp::create(rewriter, loc, 0, 32);

      for (int32_t slot = 0; slot < alloc.getNumBarriers(); ++slot) {
        Value barId =
            arith::ConstantIntOp::create(rewriter, loc, nextBase + slot, 32);
        lastInitArrival = triton::musa::InitArrivalOp::create(
            rewriter, loc, barId, arriveCount, initPolarity);
      }

      rewriter.replaceAllUsesWith(alloc.getBaseId(), base);
      rewriter.eraseOp(alloc);
      nextBase += alloc.getNumBarriers();
    }

    assert(lastInitArrival && "positive barrier allocations must initialize");
    rewriter.setInsertionPointAfter(lastInitArrival);
    ttg::BarrierOp::create(rewriter, lastInitArrival->getLoc(),
                           ttg::AddrSpace::Local);

    for (BarrierIndexOp index : indices) {
      if (!index)
        continue;
      rewriter.setInsertionPoint(index);
      APInt baseValue;
      APInt indexValue;
      Value physicalId;
      if (matchPattern(index.getBaseId(), m_ConstantInt(&baseValue)) &&
          matchPattern(index.getIndex(), m_ConstantInt(&indexValue))) {
        physicalId = arith::ConstantIntOp::create(
            rewriter, index.getLoc(),
            baseValue.getSExtValue() + indexValue.getSExtValue(), 32);
      } else {
        physicalId = arith::AddIOp::create(rewriter, index.getLoc(),
                                           index.getBaseId(), index.getIndex());
      }
      rewriter.replaceOp(index, physicalId);
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
      if (failed(lowerFunction(func, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
