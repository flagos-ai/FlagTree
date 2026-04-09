// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
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

// flagtree tle

#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEINSERTLOCALPOINTERBARRIERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kBarrierGroupAttr = "tle.barrier_group";

class InsertLocalPointerBarriersPass
    : public impl::TritonTleInsertLocalPointerBarriersBase<
          InsertLocalPointerBarriersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    pointerGroups.clear();
    collectTrackedPointers(module);

    if (pointerGroups.empty())
      return;

    for (Operation &op : module.getBody()->getOperations())
      processOperation(op);
  }

  void collectTrackedPointers(ModuleOp module) {
    llvm::SmallVector<Value> worklist;
    module.walk([&](tle::LocalPointersOp op) {
      auto groupAttr = op->getAttrOfType<IntegerAttr>(kBarrierGroupAttr);
      if (!groupAttr)
        return;
      Value ptr = op.getResult();
      int64_t group = groupAttr.getInt();
      if (pointerGroups.try_emplace(ptr, group).second)
        worklist.push_back(ptr);
    });

    auto tryTrackDerived = [&](Operation *op, Value src, Value derived) {
      auto it = pointerGroups.find(src);
      if (it == pointerGroups.end())
        return;
      if (pointerGroups.try_emplace(derived, it->second).second)
        worklist.push_back(derived);
    };

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      for (OpOperand &use : current.getUses()) {
        Operation *owner = use.getOwner();
        if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(owner)) {
          tryTrackDerived(owner, convert.getSrc(), convert.getResult());
        } else if (auto splat = dyn_cast<triton::SplatOp>(owner)) {
          tryTrackDerived(owner, splat.getSrc(), splat.getResult());
        } else if (auto bcast = dyn_cast<triton::BroadcastOp>(owner)) {
          tryTrackDerived(owner, bcast.getSrc(), bcast.getResult());
        } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(owner)) {
          tryTrackDerived(owner, expand.getSrc(), expand.getResult());
        } else if (auto reshape = dyn_cast<triton::ReshapeOp>(owner)) {
          tryTrackDerived(owner, reshape.getSrc(), reshape.getResult());
        } else if (auto addptr = dyn_cast<triton::AddPtrOp>(owner)) {
          // Only propagate along the pointer operand.
          if (use.getOperandNumber() == 0)
            tryTrackDerived(owner, addptr.getPtr(), addptr.getResult());
        } else if (auto call = dyn_cast<triton::CallOp>(owner)) {
          auto it = pointerGroups.find(current);
          if (it == pointerGroups.end())
            continue;
          unsigned operandIdx = use.getOperandNumber();
          auto callee = module.lookupSymbol<triton::FuncOp>(call.getCallee());
          if (!callee || operandIdx >= callee.getNumArguments())
            continue;
          Value calleeArg = callee.getArgument(operandIdx);
          if (pointerGroups.try_emplace(calleeArg, it->second).second)
            worklist.push_back(calleeArg);
        }
      }
    }
  }

  void processOperation(Operation &op) {
    for (Region &region : op.getRegions())
      processRegion(region);
  }

  void processRegion(Region &region) {
    for (Block &block : region)
      processBlock(block);
  }

  void processBlock(Block &block) {
    llvm::DenseMap<int64_t, bool> dirtyGroups;
    for (Operation &op : block) {
      if (!dirtyGroups.empty() && op.getNumRegions() > 0) {
        bool handledByIfSpecialization = false;
        if (auto ifOp = dyn_cast<scf::IfOp>(&op))
          handledByIfSpecialization = tryHandleUniformIf(ifOp, dirtyGroups);

        if (!handledByIfSpecialization &&
            opHasLoadNeedingBarrier(op, dirtyGroups)) {
          OpBuilder builder(&op);
          builder.create<mlir::gpu::BarrierOp>(op.getLoc());
          dirtyGroups.clear();
        }
      }

      if (auto store = dyn_cast<triton::StoreOp>(&op)) {
        if (auto group = lookupPointerGroup(store.getPtr()))
          dirtyGroups[*group] = true;
      } else if (auto load = dyn_cast<triton::LoadOp>(&op)) {
        auto group = lookupPointerGroup(load.getPtr());
        if (!group || !dirtyGroups.lookup(*group))
          continue;
        OpBuilder builder(load);
        builder.create<mlir::gpu::BarrierOp>(load.getLoc());
        // A CTA barrier synchronizes all shared-memory groups, not only the
        // group used by this load. Clearing all dirty groups avoids emitting
        // redundant back-to-back barriers for consecutive loads from different
        // tracked groups.
        dirtyGroups.clear();
      } else if (isa<mlir::gpu::BarrierOp>(&op)) {
        dirtyGroups.clear();
      }

      for (Region &nested : op.getRegions())
        processRegion(nested);

      // Propagate write hazards from nested regions to the parent block.
      // Without this, a store inside scf.if/scf.for may not mark parent state
      // dirty, so a subsequent outer load can miss the required barrier.
      markGroupsWrittenByNestedRegions(op, dirtyGroups);
    }
  }

  bool tryHandleUniformIf(scf::IfOp ifOp,
                          const llvm::DenseMap<int64_t, bool> &dirtyGroups) {
    if (!isUniformCondition(ifOp.getCondition()))
      return false;

    for (Region &region : ifOp->getRegions()) {
      if (!regionHasLoadNeedingBarrier(region, dirtyGroups))
        continue;
      if (region.empty() || region.front().empty())
        continue;

      Block &entry = region.front();
      if (isa<mlir::gpu::BarrierOp>(entry.front()))
        continue;

      OpBuilder builder(&entry, entry.begin());
      builder.create<mlir::gpu::BarrierOp>(ifOp.getLoc());
    }
    return true;
  }

  bool isUniformCondition(Value cond) const {
    if (isa_and_nonnull<arith::ConstantOp>(cond.getDefiningOp()))
      return true;

    auto reduce = cond.getDefiningOp<triton::ReduceOp>();
    if (!reduce || !cond.getType().isInteger(1))
      return false;

    Operation *combiner = reduce.getSingleCombiner();
    return combiner && isa<arith::OrIOp>(combiner);
  }

  bool regionHasLoadNeedingBarrier(
      Region &region, const llvm::DenseMap<int64_t, bool> &dirtyGroups) const {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (auto load = dyn_cast<triton::LoadOp>(&nestedOp)) {
          if (auto group = lookupPointerGroup(load.getPtr());
              group && dirtyGroups.lookup(*group))
            return true;
        }
        if (nestedOp.getNumRegions() > 0 &&
            opHasLoadNeedingBarrier(nestedOp, dirtyGroups))
          return true;
      }
    }
    return false;
  }

  bool opHasLoadNeedingBarrier(
      Operation &op, const llvm::DenseMap<int64_t, bool> &dirtyGroups) const {
    for (Region &region : op.getRegions()) {
      if (regionHasLoadNeedingBarrier(region, dirtyGroups))
        return true;
    }
    return false;
  }

  void markGroupsWrittenByNestedRegions(
      Operation &op, llvm::DenseMap<int64_t, bool> &dirtyGroups) const {
    if (op.getNumRegions() == 0)
      return;
    llvm::DenseSet<int64_t> writtenGroups;
    for (Region &region : op.getRegions())
      collectWrittenGroups(region, writtenGroups);
    for (int64_t group : writtenGroups)
      dirtyGroups[group] = true;
  }

  void collectWrittenGroups(Region &region,
                            llvm::DenseSet<int64_t> &writtenGroups) const {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (auto store = dyn_cast<triton::StoreOp>(&nestedOp)) {
          if (auto group = lookupPointerGroup(store.getPtr()))
            writtenGroups.insert(*group);
        }
        for (Region &deeperRegion : nestedOp.getRegions())
          collectWrittenGroups(deeperRegion, writtenGroups);
      }
    }
  }

  std::optional<int64_t> lookupPointerGroup(Value ptr) const {
    auto it = pointerGroups.find(ptr);
    if (it == pointerGroups.end())
      return std::nullopt;
    return it->second;
  }

  llvm::DenseMap<Value, int64_t> pointerGroups;
};

} // namespace
} // namespace mlir::triton::tle
