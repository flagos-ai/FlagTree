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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
        } else if (auto bcast = dyn_cast<triton::BroadcastOp>(owner)) {
          tryTrackDerived(owner, bcast.getSrc(), bcast.getResult());
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
      if (!dirtyGroups.empty() && op.getNumRegions() > 0 &&
          opHasLoadNeedingBarrier(op, dirtyGroups)) {
        OpBuilder builder(&op);
        builder.create<mlir::gpu::BarrierOp>(op.getLoc());
        dirtyGroups.clear();
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
        dirtyGroups[*group] = false;
      } else if (isa<mlir::gpu::BarrierOp>(&op)) {
        dirtyGroups.clear();
      }

      for (Region &nested : op.getRegions())
        processRegion(nested);
    }
  }

  bool opHasLoadNeedingBarrier(
      Operation &op, const llvm::DenseMap<int64_t, bool> &dirtyGroups) const {
    bool needsBarrier = false;
    for (Region &region : op.getRegions()) {
      for (Block &block : region) {
        for (Operation &nestedOp : block) {
          if (auto load = dyn_cast<triton::LoadOp>(&nestedOp)) {
            if (auto group = lookupPointerGroup(load.getPtr());
                group && dirtyGroups.lookup(*group)) {
              needsBarrier = true;
              break;
            }
          }
          if (nestedOp.getNumRegions() > 0 &&
              opHasLoadNeedingBarrier(nestedOp, dirtyGroups)) {
            needsBarrier = true;
            break;
          }
        }
        if (needsBarrier)
          break;
      }
      if (needsBarrier)
        break;
    }
    return needsBarrier;
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
