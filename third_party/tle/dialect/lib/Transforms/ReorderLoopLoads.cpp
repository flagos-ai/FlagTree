#include "tle/dialect/include/Transforms/ReorderLoopLoads.h"

#include "tle/dialect/include/Transforms/TransformAttrs.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tle-reorder-loop-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::tle {

std::optional<int64_t> getReorderLoopLoadsDistance(scf::ForOp forOp) {
  Attribute attr = forOp->getAttr(kTleReorderLoopLoadsAttr);
  // BoolAttr is an i1 IntegerAttr, so it has to be tested first.
  if (auto boolAttr = dyn_cast_or_null<BoolAttr>(attr))
    return boolAttr.getValue() ? std::optional<int64_t>(-1) : std::nullopt;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr)) {
    int64_t distance = intAttr.getInt();
    return distance >= 1 ? std::optional<int64_t>(distance) : std::nullopt;
  }
  return std::nullopt;
}

void removeReorderLoopLoadsAttr(scf::ForOp forOp) {
  forOp->removeAttr(kTleReorderLoopLoadsAttr);
}

void annotateUnrolledIteration(unsigned iteration, Operation *op, OpBuilder b) {
  op->setAttr(kTleReorderIterationAttr, b.getI32IntegerAttr(iteration));
}

namespace {

bool isNonVolatileLoad(Operation *op) {
  auto load = dyn_cast<triton::LoadOp>(op);
  return load && !load.getIsVolatile();
}

// Whether `op` only reads memory (or has no memory effect at all).
bool onlyReadsMemory(Operation *op) {
  if (isMemoryEffectFree(op))
    return true;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  return llvm::all_of(effects, [](const MemoryEffects::EffectInstance &e) {
    return isa<MemoryEffects::Read>(e.getEffect());
  });
}

// A barrier (tl.debug_barrier) orders the loads and stores of the iteration it
// sits in: a load after it stays after it (see the scan in reorderRange). It
// is not a fence for the loads of later iterations, which by the hint's
// contract do not depend on what earlier iterations store. Matched by name so
// that this file does not depend on the GPU dialect.
bool isBarrier(Operation *op) {
  return op->getName().getStringRef() == "gpu.barrier";
}

// Whether a load of a later iteration must not be issued ahead of `op`.
// Stores are deliberately not fences: reordering a load with the stores of
// *other* unrolled iterations is what the `reorder` hint asks for (the user
// asserts that they do not alias), and neither are barriers (see isBarrier).
// Everything else that writes memory or has unknown effects is a fence, as is
// a volatile load.
bool isReorderFence(Operation *op) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return load.getIsVolatile();
  if (isa<triton::StoreOp>(op) || isBarrier(op))
    return false;
  if (onlyReadsMemory(op))
    return false;
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    WalkResult result = op->walk([&](Operation *nested) {
      if (nested != op && isReorderFence(nested))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return result.wasInterrupted();
  }
  return true;
}

// Whether `op` writes memory in a way a later load of the same iteration may
// observe (stores and fences).
bool mayWriteMemory(Operation *op) {
  if (isa<triton::StoreOp>(op))
    return true;
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
      !onlyReadsMemory(op))
    return true;
  return isReorderFence(op);
}

// Ops in [begin, end) of `block` that define a value `op` (or an op nested in
// it) uses.
void forEachInRangeProducer(Operation *op, const DenseSet<Operation *> &range,
                            function_ref<void(Operation *)> fn) {
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && def != op && range.contains(def))
        fn(def);
    }
  });
}

void reorderRange(Block *block, Operation *beginOp, Operation *endOp,
                  int64_t distance) {
  SmallVector<Operation *> ops;
  for (auto it = Block::iterator(beginOp); &*it != endOp; ++it)
    ops.push_back(&*it);

  // Iteration of every op. Ops loopUnrollByFactor created without a tag (the
  // remapped induction variable of each copy) belong to the next tagged op;
  // untagged ops after the last copy form the tail.
  int64_t numIterations = 0;
  for (Operation *op : ops)
    if (auto attr = op->getAttrOfType<IntegerAttr>(kTleReorderIterationAttr))
      numIterations = std::max<int64_t>(numIterations, attr.getInt() + 1);
  if (numIterations <= 1)
    return;
  const int64_t tail = numIterations;
  DenseMap<Operation *, int64_t> iterOf;
  int64_t next = tail;
  for (Operation *op : llvm::reverse(ops)) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kTleReorderIterationAttr))
      next = attr.getInt();
    iterOf[op] = next;
  }

  DenseSet<Operation *> inRange(ops.begin(), ops.end());
  DenseMap<Operation *, unsigned> position;
  for (auto [i, op] : llvm::enumerate(ops))
    position[op] = i;

  // A load is hoisted together with the producers of its operands. That is
  // only legal when it precedes every write and every barrier of its own
  // iteration and every producer is free of side effects or a hoisted load
  // itself.
  DenseSet<Operation *> hoistable;
  DenseMap<int64_t, SmallVector<Operation *>> loadsOf;
  SmallVector<bool> wroteMemory(numIterations + 1, false);
  SmallVector<bool> sawBarrier(numIterations + 1, false);
  SmallVector<bool> hasFence(numIterations + 1, false);
  for (Operation *op : ops) {
    int64_t iter = iterOf[op];
    if (isNonVolatileLoad(op) && iter < tail && !wroteMemory[iter] &&
        !sawBarrier[iter]) {
      bool movable = true;
      SmallVector<Operation *> worklist{op};
      DenseSet<Operation *> seen{op};
      while (!worklist.empty() && movable) {
        Operation *cur = worklist.pop_back_val();
        forEachInRangeProducer(cur, inRange, [&](Operation *def) {
          if (!seen.insert(def).second)
            return;
          if (isNonVolatileLoad(def) ? !hoistable.contains(def)
                                     : !isMemoryEffectFree(def))
            movable = false;
          worklist.push_back(def);
        });
      }
      if (movable) {
        hoistable.insert(op);
        loadsOf[iter].push_back(op);
        continue;
      }
    }
    if (mayWriteMemory(op))
      wroteMemory[iter] = true;
    if (isBarrier(op))
      sawBarrier[iter] = true;
    if (isReorderFence(op))
      hasFence[iter] = true;
  }
  if (hoistable.empty()) {
    LDBG("No hoistable loads in the unrolled range");
    return;
  }

  // Emit every op exactly once: a load group pulls in the producers it needs
  // (in their original order), the compute of iteration c takes whatever of
  // iteration c is left.
  DenseSet<Operation *> emitted;
  SmallVector<Operation *> order;
  auto emitLoadsOf = [&](int64_t iter) {
    SetVector<Operation *> closure;
    SmallVector<Operation *> worklist(loadsOf[iter].begin(),
                                      loadsOf[iter].end());
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      if (emitted.contains(cur) || !closure.insert(cur))
        continue;
      forEachInRangeProducer(cur, inRange,
                             [&](Operation *def) { worklist.push_back(def); });
    }
    SmallVector<Operation *> sorted(closure.begin(), closure.end());
    llvm::sort(sorted, [&](Operation *a, Operation *b) {
      return position[a] < position[b];
    });
    for (Operation *op : sorted)
      if (emitted.insert(op).second)
        order.push_back(op);
  };
  auto emitRestOf = [&](int64_t iter) {
    for (Operation *op : ops)
      if (iterOf[op] == iter && emitted.insert(op).second)
        order.push_back(op);
  };

  const int64_t ahead =
      distance < 0 ? numIterations - 1 : std::min(distance, numIterations - 1);
  int64_t nextLoads = 0;
  for (int64_t c = 0; c < numIterations; ++c) {
    // Issue the loads of iterations up to c + ahead, but never ahead of a
    // fence in the compute of iterations c .. nextLoads - 1.
    while (nextLoads < numIterations && nextLoads <= c + ahead) {
      bool blocked = false;
      for (int64_t j = c; j < nextLoads; ++j)
        blocked |= hasFence[j];
      if (blocked)
        break;
      emitLoadsOf(nextLoads++);
    }
    emitRestOf(c);
  }
  emitRestOf(tail);
  assert(order.size() == ops.size() && "every op is emitted once");

  LDBG("Reordered " << numIterations << " unrolled iterations, "
                    << hoistable.size() << " hoistable loads, distance "
                    << ahead);
  for (Operation *op : order)
    op->moveBefore(endOp);
}

} // namespace

void reorderLoopLoadsAfterUnroll(Block *parentBlock, Operation *opBeforeLoop,
                                 Operation *opAfterLoop, int64_t distance) {
  auto rangeBegin = [&]() {
    return opBeforeLoop ? opBeforeLoop->getNextNode() : &parentBlock->front();
  };
  if (rangeBegin() == opAfterLoop)
    return;
  // Fully unrolled: the copies were inlined into the parent block.
  reorderRange(parentBlock, rangeBegin(), opAfterLoop, distance);
  // Partially unrolled: the copies live in the body of the main loop.
  for (auto it = Block::iterator(rangeBegin());
       it != Block::iterator(opAfterLoop); ++it) {
    if (auto forOp = dyn_cast<scf::ForOp>(&*it)) {
      Block *body = forOp.getBody();
      if (&body->front() != body->getTerminator())
        reorderRange(body, &body->front(), body->getTerminator(), distance);
    }
  }
  for (auto it = Block::iterator(rangeBegin());
       it != Block::iterator(opAfterLoop); ++it)
    it->walk([](Operation *op) { op->removeAttr(kTleReorderIterationAttr); });
}

} // namespace mlir::triton::tle
