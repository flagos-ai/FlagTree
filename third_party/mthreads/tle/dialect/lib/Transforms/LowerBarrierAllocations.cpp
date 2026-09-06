#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
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
static constexpr StringLiteral kPipeExhaustedDiagnostic =
    "MUSA TLE pipe barrier allocation exceeds hardware barrier id limit";
static constexpr StringLiteral kPipeBarrierRingAttr =
    "musa_tle.pipe_barrier_ring";

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

    BarrierAllocOp pipeAlloc;
    for (BarrierAllocOp alloc : allocs) {
      if (alloc->hasAttr(kPipeBarrierRingAttr)) {
        pipeAlloc = alloc;
        break;
      }
    }

    if (totalBarriers <= 0 ||
        totalBarriers > std::numeric_limits<int32_t>::max()) {
      if (pipeAlloc)
        pipeAlloc.emitOpError(kPipeExhaustedDiagnostic);
      else
        allocs.front().emitOpError()
            << kExhaustedDiagnostic << ": cannot reserve " << totalBarriers
            << " additional ids in [1, " << triton::musa::kMaxBarrierId << "]";
      return failure();
    }

    auto reserved = triton::musa::reserveBarrierIdRange(
        allocs.front(), static_cast<int32_t>(totalBarriers));
    if (failed(reserved)) {
      if (pipeAlloc)
        pipeAlloc.emitOpError(kPipeExhaustedDiagnostic);
      else
        allocs.front().emitOpError()
            << kExhaustedDiagnostic << ": cannot reserve " << totalBarriers
            << " additional ids in [1, " << triton::musa::kMaxBarrierId << "]";
      return failure();
    }

    SmallVector<BarrierIndexOp> indices;
    func.walk([&](BarrierIndexOp op) { indices.push_back(op); });

    Operation *lastInitArrival = nullptr;
    int32_t nextBase = *reserved;
    SmallVector<Value> loweredBases;
    for (BarrierAllocOp alloc : allocs) {
      Location loc = alloc.getLoc();
      rewriter.setInsertionPoint(alloc);
      Value base = arith::ConstantIntOp::create(rewriter, loc, nextBase, 32);
      loweredBases.push_back(base);
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
      alloc->removeAttr(kPipeBarrierRingAttr);
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

    // A hardware barrier is identified by an i32 resource ID and does not
    // require shared-memory storage. Pipe lowering may need to carry the base
    // ID into an isolated warp-specialize producer region before IDs are
    // assigned. Once allocation has resolved the base to an arith.constant,
    // rematerialize that constant in each partition and remove only those
    // barrier-ID captures. This prevents the generic WS capture mailbox from
    // charging shared memory for hardware barrier IDs.
    SmallVector<ttg::WarpSpecializePartitionsOp> partitionContainers;
    func.walk([&](ttg::WarpSpecializePartitionsOp partitions) {
      partitionContainers.push_back(partitions);
    });
    for (ttg::WarpSpecializePartitionsOp partitions : partitionContainers) {
      llvm::BitVector eraseCaptures(partitions.getNumOperands());
      for (auto [index, capture] :
           llvm::enumerate(partitions.getExplicitCaptures())) {
        if (!llvm::is_contained(loweredBases, capture))
          continue;
        Operation *constant = capture.getDefiningOp();
        assert(isa_and_nonnull<arith::ConstantIntOp>(constant));
        for (Region &partition : partitions.getPartitionRegions()) {
          rewriter.setInsertionPointToStart(&partition.front());
          IRMapping mapping;
          Operation *clone = rewriter.clone(*constant, mapping);
          partition.getArgument(index).replaceAllUsesWith(clone->getResult(0));
        }
        eraseCaptures.set(index);
      }
      if (eraseCaptures.none())
        continue;
      for (Region &partition : partitions.getPartitionRegions())
        partition.front().eraseArguments(eraseCaptures);
      partitions->eraseOperands(eraseCaptures);
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
