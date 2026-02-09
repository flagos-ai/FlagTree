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
    trackedPointers.clear();
    collectTrackedPointers(module);

    if (trackedPointers.empty())
      return;

    for (Operation &op : module.getBody()->getOperations())
      processOperation(op);
  }

  void collectTrackedPointers(ModuleOp module) {
    llvm::SmallVector<Value> worklist;
    module.walk([&](tle::LocalPointersOp op) {
      if (!op->hasAttr(kBarrierGroupAttr))
        return;
      Value ptr = op.getResult();
      if (trackedPointers.insert(ptr).second)
        worklist.push_back(ptr);
    });

    auto tryTrackDerived = [&](Operation *op, Value src, Value derived) {
      if (!trackedPointers.contains(src))
        return;
      if (trackedPointers.insert(derived).second)
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
    llvm::DenseMap<Value, bool> dirty;
    for (Operation &op : block) {
      if (auto store = dyn_cast<triton::StoreOp>(&op)) {
        Value ptr = store.getPtr();
        if (trackedPointers.contains(ptr))
          dirty[ptr] = true;
      } else if (auto load = dyn_cast<triton::LoadOp>(&op)) {
        Value ptr = load.getPtr();
        if (!trackedPointers.contains(ptr) || !dirty.lookup(ptr))
          continue;
        OpBuilder builder(load);
        builder.create<mlir::gpu::BarrierOp>(load.getLoc());
        dirty[ptr] = false;
      } else if (isa<mlir::gpu::BarrierOp>(&op)) {
        dirty.clear();
      }

      for (Region &nested : op.getRegions())
        processRegion(nested);
    }
  }

  llvm::DenseSet<Value> trackedPointers;
};

} // namespace
} // namespace mlir::triton::tle
