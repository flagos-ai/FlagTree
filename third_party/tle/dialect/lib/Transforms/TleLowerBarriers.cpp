/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLELOWERBARRIERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static int64_t getI32Attr(Operation *op, StringRef name) {
  return op->getAttrOfType<IntegerAttr>(name).getInt();
}

static ttg::MemDescType getBarrierSlotType(ttg::MemDescType arrayTy) {
  auto context = arrayTy.getContext();
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  Attribute slotEncoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
  return ttg::MemDescType::get({1}, arrayTy.getElementType(), slotEncoding,
                               arrayTy.getMemorySpace(),
                               arrayTy.getMutableMemory());
}

static Value createBarrierSlot(OpBuilder &builder, Location loc, Value array,
                               int64_t index) {
  auto arrayTy = cast<ttg::MemDescType>(array.getType());
  auto slotTy = getBarrierSlotType(arrayTy);
  Value idx = builder.create<arith::ConstantIntOp>(loc, index, 32);
  return builder.create<ttg::MemDescIndexOp>(loc, slotTy, array, idx);
}

// Trace a barrier slot passed into an isolated warp-specialization partition
// back to its source allocation.  The partition capture is deliberately still
// present when this pass runs; canonicalization removes unused captures only
// after named barrier ops have been lowered.
static BarrierAllocOp findBarrierAlloc(Value value) {
  while (value) {
    if (auto view = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = view.getSrc();
      continue;
    }
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return value.getDefiningOp<BarrierAllocOp>();
    auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
        blockArg.getOwner()->getParentOp());
    if (!partitions)
      return {};
    auto warpSpecialize =
        dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!warpSpecialize)
      return {};
    OperandRange captures = warpSpecialize.getExplicitCaptures();
    if (blockArg.getArgNumber() >= captures.size())
      return {};
    value = captures[blockArg.getArgNumber()];
  }
  return {};
}

#if !defined(__HCU__)
static std::pair<Value, Value>
createNamedBarrierOperands(OpBuilder &builder, Location loc, Operation *op) {
  Value id =
      builder.create<arith::ConstantIntOp>(loc, getI32Attr(op, "named_id"), 32);
  Value threads = builder.create<arith::ConstantIntOp>(
      loc, getI32Attr(op, "named_num_threads"), 32);
  return {id, threads};
}
#endif

struct TritonTleLowerBarriers
    : public impl::TritonTleLowerBarriersBase<TritonTleLowerBarriers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<BarrierWaitOp> waits;
    SmallVector<BarrierArriveOp> arrives;
    SmallVector<BarrierAllocOp> allocs;
    llvm::MapVector<Value, SmallVector<Value>> ownedBarrierSlots;
    module.walk([&](Operation *op) {
      if (auto wait = dyn_cast<BarrierWaitOp>(op))
        waits.push_back(wait);
      else if (auto arrive = dyn_cast<BarrierArriveOp>(op))
        arrives.push_back(arrive);
      else if (auto alloc = dyn_cast<BarrierAllocOp>(op))
        allocs.push_back(alloc);
    });

    // Named barriers are hardware resources identified only by an integer id;
    // their memdesc operands are frontend SSA handles, not shared-memory
    // storage.  Remember which source allocations are named-only before
    // erasing the wait/arrive ops.  Otherwise a capture through
    // ttg.warp_specialize keeps the handle artificially live and the generic
    // allocation lowering emits an mbarrier.init into unrelated shared memory.
    llvm::DenseSet<Operation *> namedAllocs;
    llvm::DenseSet<Operation *> mbarrierAllocs;
    auto recordBackend = [&](Operation *op, Value barrier) {
      auto alloc = findBarrierAlloc(barrier);
      if (!alloc)
        return;
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      (backend == "named" ? namedAllocs : mbarrierAllocs)
          .insert(alloc.getOperation());
    };
    for (BarrierWaitOp op : waits)
      recordBackend(op.getOperation(), op.getBarrier());
    for (BarrierArriveOp op : arrives)
      recordBackend(op.getOperation(), op.getBarrier());

    for (BarrierWaitOp op : waits) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        builder.create<ttng::WaitBarrierOp>(loc, op.getBarrier(),
                                            op.getPhase());
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierWaitOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierArriveOp op : arrives) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        int64_t count = getI32Attr(op.getOperation(), "arrive_count");
        builder.create<ttng::ArriveBarrierOp>(loc, op.getBarrier(),
                                              static_cast<uint32_t>(count));
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierArriveOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierAllocOp op : allocs) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto arrayTy = op.getResult().getType();
      bool onlyUnusedViews = true;
      SmallVector<ttg::MemDescIndexOp> deadViews;
      for (OpOperand &use : op.getResult().getUses()) {
        auto view = dyn_cast<ttg::MemDescIndexOp>(use.getOwner());
        if (!view || !view->use_empty()) {
          onlyUnusedViews = false;
          break;
        }
        deadViews.push_back(view);
      }
      if (onlyUnusedViews) {
        for (auto view : deadViews)
          view.erase();
        op.erase();
        continue;
      }

      Value alloc = builder.create<ttg::LocalAllocOp>(loc, arrayTy);
      bool namedOnly = namedAllocs.contains(op.getOperation()) &&
                       !mbarrierAllocs.contains(op.getOperation());
      int64_t numBarriers = getI32Attr(op.getOperation(), "num_barriers");
      int64_t arriveCount = getI32Attr(op.getOperation(), "arrive_count");
      if (!namedOnly) {
        for (int64_t i = 0; i < numBarriers; ++i) {
          Value slot = createBarrierSlot(builder, loc, alloc, i);
          builder.create<ttng::InitBarrierOp>(
              loc, slot, static_cast<uint32_t>(arriveCount));
          ownedBarrierSlots[alloc].push_back(slot);
        }
      }
      // InitBarrierOp is executed by one thread. Publish initialization before
      // other threads can wait on a READY barrier or start a TMA transaction.
      builder.create<mlir::gpu::BarrierOp>(loc);
      op.getResult().replaceAllUsesWith(alloc);
      op.erase();
    }

    if (ownedBarrierSlots.empty())
      return;

    // Shared allocations are local to their defining structured region. Follow
    // descriptor aliases (including loop-carried values) before closing their
    // lifetime; a borrowed descriptor never appears in ownedBarrierSlots.
    llvm::MapVector<Operation *, SmallVector<Value>> functionBarriers;
    for (const auto &[alloc, slots] : ownedBarrierSlots) {
      Operation *function =
          alloc.getDefiningOp()->getParentOfType<FunctionOpInterface>();
      if (!function ||
          !alloc.getDefiningOp()->getParentRegion()->hasOneBlock()) {
        alloc.getDefiningOp()->emitOpError(
            "owned mbarriers require a single-block defining region");
        return signalPassFailure();
      }
      functionBarriers[function].push_back(alloc);
    }
    for (const auto &[function, barriers] : functionBarriers) {
      auto solver = createDataFlowSolver();
      auto *aliases = solver->load<SharedMemoryAliasAnalysis>();
      if (failed(solver->initializeAndRun(function)))
        return signalPassFailure();
      DenseSet<Value> escaped;
      auto checkValue = [&](Value value) {
        auto *lattice = aliases->getLatticeElement(value);
        for (Value alloc : barriers) {
          if (escaped.contains(alloc) ||
              !lattice->getValue().getAllocs().contains(alloc))
            continue;
          Region *scope = alloc.getParentRegion();
          bool outsideScope = !scope->isAncestor(value.getParentRegion());
          bool returnedFromScope =
              llvm::any_of(value.getUsers(), [&](Operation *user) {
                return user == alloc.getParentBlock()->getTerminator();
              });
          if (outsideScope || returnedFromScope) {
            alloc.getDefiningOp()->emitOpError(
                "owned mbarrier cannot escape its defining region; allocate "
                "the barrier in the caller and pass it as a borrowed argument");
            escaped.insert(alloc);
          }
        }
      };
      function->walk([&](Operation *operation) {
        for (Value value : operation->getResults())
          checkValue(value);
        for (Region &region : operation->getRegions())
          for (Block &block : region)
            for (Value value : block.getArguments())
              checkValue(value);
      });
      if (!escaped.empty())
        return signalPassFailure();
    }

    llvm::MapVector<Block *, SmallVector<Value>> scopeBarrierSlots;
    for (const auto &[alloc, slots] : ownedBarrierSlots)
      scopeBarrierSlots[alloc.getDefiningOp()->getBlock()].append(slots);
    for (const auto &[block, slots] : scopeBarrierSlots) {
      OpBuilder builder(block->getTerminator());
      Location loc = block->getTerminator()->getLoc();
      // All users must finish before thread zero invalidates the hardware
      // objects. All threads must observe invalidation before the allocator
      // can reuse these bytes for a later helper's unrelated shared storage.
      builder.create<mlir::gpu::BarrierOp>(loc);
      for (Value slot : slots)
        builder.create<ttng::InvalBarrierOp>(loc, slot);
      builder.create<mlir::gpu::BarrierOp>(loc);
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
