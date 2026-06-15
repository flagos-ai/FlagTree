// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttnvws = mlir::triton::nvws;

#define GEN_PASS_DEF_TRITONTLESCHEDULETMASTORESYNC
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

struct PendingTMAStoreGroup {
  SmallVector<Value, 2> sourceRoots;
};

static Value canonicalizeWarpSpecializeCapture(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions)
      break;
    auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!wsOp)
      break;
    unsigned argNo = blockArg.getArgNumber();
    OperandRange captures = wsOp.getExplicitCaptures();
    if (argNo >= captures.size())
      break;
    value = captures[argNo];
  }
  return value;
}

static Value getMemDescRoot(Value value) {
  Value current = canonicalizeWarpSpecializeCapture(value);
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = canonicalizeWarpSpecializeCapture(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizeWarpSpecializeCapture(subslice.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = canonicalizeWarpSpecializeCapture(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = canonicalizeWarpSpecializeCapture(reshape.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = canonicalizeWarpSpecializeCapture(reinterpret.getSrc());
      continue;
    }
    if (auto wgmmaView = current.getDefiningOp<MemDescWGMMAViewOp>()) {
      current = canonicalizeWarpSpecializeCapture(wgmmaView.getSrc());
      continue;
    }
    break;
  }
  return current;
}

static void appendUniqueRoot(SmallVectorImpl<Value> &roots, Value root) {
  if (!llvm::is_contained(roots, root))
    roots.push_back(root);
}

static std::optional<unsigned>
findLatestAliasingGroup(ArrayRef<PendingTMAStoreGroup> pendingGroups,
                        Value value) {
  Value root = getMemDescRoot(value);
  std::optional<unsigned> latest;
  for (auto indexed : llvm::enumerate(pendingGroups)) {
    if (llvm::is_contained(indexed.value().sourceRoots, root))
      latest = indexed.index();
  }
  return latest;
}

static std::optional<unsigned> mergeLatest(std::optional<unsigned> lhs,
                                           std::optional<unsigned> rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return std::max(*lhs, *rhs);
}

static bool isTLEExplicitTMAStore(ttng::AsyncTMACopyLocalToGlobalOp op) {
  return op->hasAttr(kTleTMAStoreExplicitCommitAttr);
}

static bool isNonTLEStoreGroupBoundary(Operation *op) {
  if (auto tmaStore = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return !isTLEExplicitTMAStore(tmaStore);
  return isa<ttng::AsyncTMAReduceOp, ttng::AsyncTMAScatterOp>(op);
}

static bool isPassThroughForTLEStoreRun(Operation *op) {
  return isa<ttng::FenceAsyncSharedOp>(op);
}

static bool isWriteOrFree(MemoryEffects::Effect *effect) {
  return isa<MemoryEffects::Write, MemoryEffects::Free>(effect);
}

static std::optional<unsigned>
findMemoryReuseHazard(Operation *op,
                      ArrayRef<PendingTMAStoreGroup> pendingGroups) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return std::nullopt;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  effectInterface.getEffects(effects);

  std::optional<unsigned> latestHazard;
  for (const auto &effect : effects) {
    if (!isWriteOrFree(effect.getEffect()))
      continue;
    Value value = effect.getValue();
    if (!value || !isa<ttg::MemDescType>(value.getType()))
      continue;
    latestHazard = mergeLatest(latestHazard,
                               findLatestAliasingGroup(pendingGroups, value));
  }
  return latestHazard;
}

static std::optional<unsigned>
findPipeReaderReleaseHazard(Operation *op,
                            ArrayRef<PendingTMAStoreGroup> pendingGroups) {
  auto release = dyn_cast<PipeReaderReleaseOp>(op);
  if (!release)
    return std::nullopt;

  std::optional<unsigned> latestHazard;
  for (Value field : release.getFields()) {
    latestHazard = mergeLatest(latestHazard,
                               findLatestAliasingGroup(pendingGroups, field));
  }
  return latestHazard;
}

static void insertWaitBefore(OpBuilder &builder, Operation *op,
                             unsigned pendings) {
  builder.setInsertionPoint(op);
  ttng::TMAStoreWaitOp::create(builder, op->getLoc(), pendings);
}

static bool
waitThroughGroupBefore(OpBuilder &builder, Operation *op,
                       SmallVectorImpl<PendingTMAStoreGroup> &pendingGroups,
                       unsigned groupIndex) {
  assert(groupIndex < pendingGroups.size());
  unsigned pendings = pendingGroups.size() - groupIndex - 1;
  insertWaitBefore(builder, op, pendings);
  pendingGroups.erase(pendingGroups.begin(),
                      pendingGroups.begin() + groupIndex + 1);
  return true;
}

static bool
waitAllBefore(OpBuilder &builder, Operation *op,
              SmallVectorImpl<PendingTMAStoreGroup> &pendingGroups) {
  if (pendingGroups.empty())
    return false;
  insertWaitBefore(builder, op, 0);
  pendingGroups.clear();
  return true;
}

static bool
commitCurrentGroupBefore(OpBuilder &builder, Operation *op,
                         SmallVectorImpl<Value> &currentRoots,
                         SmallVectorImpl<PendingTMAStoreGroup> &pendingGroups) {
  if (currentRoots.empty())
    return false;

  builder.setInsertionPoint(op);
  TMAStoreCommitGroupOp::create(builder, op->getLoc());
  PendingTMAStoreGroup group;
  for (Value root : currentRoots)
    group.sourceRoots.push_back(root);
  pendingGroups.push_back(std::move(group));
  currentRoots.clear();
  return true;
}

static bool
commitCurrentGroupAtEnd(OpBuilder &builder, Block &block,
                        SmallVectorImpl<Value> &currentRoots,
                        SmallVectorImpl<PendingTMAStoreGroup> &pendingGroups) {
  if (currentRoots.empty())
    return false;

  Location loc =
      block.empty() ? builder.getUnknownLoc() : block.back().getLoc();
  builder.setInsertionPointToEnd(&block);
  TMAStoreCommitGroupOp::create(builder, loc);
  PendingTMAStoreGroup group;
  for (Value root : currentRoots)
    group.sourceRoots.push_back(root);
  pendingGroups.push_back(std::move(group));
  currentRoots.clear();
  return true;
}

static bool waitAllAtEnd(OpBuilder &builder, Block &block,
                         SmallVectorImpl<PendingTMAStoreGroup> &pendingGroups) {
  if (pendingGroups.empty())
    return false;

  Location loc =
      block.empty() ? builder.getUnknownLoc() : block.back().getLoc();
  builder.setInsertionPointToEnd(&block);
  ttng::TMAStoreWaitOp::create(builder, loc, 0);
  pendingGroups.clear();
  return true;
}

static bool scheduleBlock(Block &block) {
  OpBuilder builder(block.getParentOp());
  SmallVector<Value, 2> currentRoots;
  SmallVector<PendingTMAStoreGroup, 4> pendingGroups;
  bool changed = false;

  for (auto it = block.begin(), end = block.end(); it != end;) {
    Operation *op = &*it++;

    if (auto tmaStore = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      if (isTLEExplicitTMAStore(tmaStore)) {
        appendUniqueRoot(currentRoots, getMemDescRoot(tmaStore.getSrc()));
        continue;
      }

      changed |=
          commitCurrentGroupBefore(builder, op, currentRoots, pendingGroups);
      changed |= waitAllBefore(builder, op, pendingGroups);
      continue;
    }

    if (auto commit = dyn_cast<TMAStoreCommitGroupOp>(op)) {
      if (!currentRoots.empty()) {
        commit.erase();
        changed = true;
      }
      continue;
    }

    if (auto wait = dyn_cast<ttng::TMAStoreWaitOp>(op)) {
      if (!currentRoots.empty() || !pendingGroups.empty()) {
        wait.erase();
        changed = true;
      }
      continue;
    }

    if (!currentRoots.empty() && isPassThroughForTLEStoreRun(op))
      continue;

    if (isNonTLEStoreGroupBoundary(op)) {
      changed |=
          commitCurrentGroupBefore(builder, op, currentRoots, pendingGroups);
      changed |= waitAllBefore(builder, op, pendingGroups);
      continue;
    }

    if (op->hasTrait<OpTrait::IsTerminator>()) {
      changed |=
          commitCurrentGroupBefore(builder, op, currentRoots, pendingGroups);
      changed |= waitAllBefore(builder, op, pendingGroups);
      continue;
    }

    changed |=
        commitCurrentGroupBefore(builder, op, currentRoots, pendingGroups);

    // At this point pipe reader releases may already be lowered to a plain
    // mbarrier arrive. Once field operands are gone, the cross-warp signal is
    // the conservative lifetime boundary for a pending TMA store source.
    if (isa<ttnvws::ConsumerReleaseOp, ttng::ArriveBarrierOp>(op)) {
      changed |= waitAllBefore(builder, op, pendingGroups);
      continue;
    }

    std::optional<unsigned> hazard =
        findPipeReaderReleaseHazard(op, pendingGroups);
    hazard = mergeLatest(hazard, findMemoryReuseHazard(op, pendingGroups));
    if (hazard)
      changed |= waitThroughGroupBefore(builder, op, pendingGroups, *hazard);
  }

  if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
    changed |=
        commitCurrentGroupAtEnd(builder, block, currentRoots, pendingGroups);
    changed |= waitAllAtEnd(builder, block, pendingGroups);
  }

  return changed;
}

class TritonTleScheduleTmaStoreSyncPass
    : public impl::TritonTleScheduleTmaStoreSyncBase<
          TritonTleScheduleTmaStoreSyncPass> {
public:
  using impl::TritonTleScheduleTmaStoreSyncBase<
      TritonTleScheduleTmaStoreSyncPass>::TritonTleScheduleTmaStoreSyncBase;

  void runOnOperation() override {
    SmallVector<Block *, 16> blocks;
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region)
          blocks.push_back(&block);
      }
    });

    for (Block *block : blocks)
      scheduleBlock(*block);
  }
};

} // namespace

} // namespace mlir::triton::tle
