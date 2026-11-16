#pragma once

#include "tle/dialect/include/Transforms/GridBarrierScratch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <functional>

namespace mlir::triton::tle {

// Validate the ordinary allocator's complete nested frame calculation in wide
// arithmetic before it writes signed-i32 offsets. No IR is changed here. This
// additional check is selected only by the explicit hierarchical protocol.
inline LogicalResult verifyHierarchicalGridScratchFrames(ModuleOp module) {
  if (failed(verifyFlagsGridModule(module)))
    return failure();
  bool selected = false;
  module.walk([&](Operation *op) {
    selected |= isHierarchicalGridBarrier(op) || isFlagsGridBarrier(op);
  });
  if (!selected)
    return success();

  struct Frame {
    uint64_t size;
    uint64_t alignment;
  };
  llvm::DenseMap<Operation *, Frame> frames;
  llvm::SmallPtrSet<Operation *, 8> active;
  constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
  std::function<FailureOr<Frame>(Operation *)> visit;
  visit = [&](Operation *parent) -> FailureOr<Frame> {
    if (auto found = frames.find(parent); found != frames.end())
      return found->second;
    if (!active.insert(parent).second) {
      parent->emitError("hierarchical grid scratch does not support recursive calls");
      return failure();
    }
    uint64_t offset = 0, largestAlignment = 1;
    bool invalid = false;
    parent->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (invalid)
        return;
      uint64_t bytes = 0, alignment = 1;
      if (auto alloc = dyn_cast<triton::gpu::GlobalScratchAllocOp>(op)) {
        bytes = alloc.getNbytes();
        alignment = alloc.getAlignment();
      } else if ((bytes = getGridBarrierScratchBytes(op)) != 0) {
        alignment = getGridBarrierScratchAlignment(op);
      } else if (auto call = dyn_cast<triton::CallOp>(op)) {
        auto callee = call.resolveCallable();
        if (!callee) {
          op->emitError("hierarchical grid scratch needs a resolved callee");
          invalid = true;
          return;
        }
        auto frame = visit(callee);
        if (failed(frame)) {
          invalid = true;
          return;
        }
        bytes = frame->size;
        alignment = frame->alignment;
      }
      if (!bytes)
        return;
      if (!llvm::isPowerOf2_64(alignment) || alignment > limit || bytes > limit) {
        op->emitError("hierarchical grid scratch size/alignment exceeds signed i32");
        invalid = true;
        return;
      }
      uint64_t aligned = llvm::alignTo(offset, alignment);
      largestAlignment = std::max(largestAlignment, alignment);
      // The existing signed roundUp also forms offset+alignment-1.
      if (aligned > limit - bytes ||
          aligned + bytes > limit - (largestAlignment - 1)) {
        op->emitError("hierarchical grid scratch accumulated frame exceeds signed i32");
        invalid = true;
        return;
      }
      offset = aligned + bytes;
    });
    active.erase(parent);
    if (invalid)
      return failure();
    Frame frame{llvm::alignTo(offset, largestAlignment), largestAlignment};
    auto priorSize = parent->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_size");
    auto priorAlignment = parent->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_alignment");
    if ((bool(priorSize) != bool(priorAlignment)) ||
        (priorSize && (priorSize.getInt() < 0 || priorAlignment.getInt() <= 0 ||
                       uint64_t(priorSize.getInt()) != frame.size ||
                       uint64_t(priorAlignment.getInt()) != frame.alignment))) {
      parent->emitError("hierarchical grid scratch has inconsistent preexisting frame metadata");
      return failure();
    }
    frames[parent] = frame;
    return frame;
  };
  bool invalid = false;
  module.walk([&](triton::FuncOp func) {
    if (!invalid && failed(visit(func)))
      invalid = true;
  });
  return failure(invalid);
}

} // namespace mlir::triton::tle
