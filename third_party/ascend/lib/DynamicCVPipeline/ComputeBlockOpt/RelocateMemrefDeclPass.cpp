/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

static constexpr const char *DEBUG_TYPE = "relocate-memref-decl";
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] " << X << "\n")

namespace mlir {
namespace triton {
namespace {

// Every operand of every member must either be relocated together with the
// cluster or already dominate the insertion point (the anchor).
static bool allOperandsDominate(const llvm::SetVector<Operation *> &cluster,
                                Operation *anchor, DominanceInfo &domInfo) {
  for (Operation *member : cluster) {
    for (Value operand : member->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || cluster.contains(def)) {
        continue; // block argument, or relocated together with the cluster
      }
      if (!domInfo.dominates(def, anchor)) {
        return false;
      }
    }
  }
  return true;
}

// True when `def`'s block strictly contains `targetBlock`, so `def` (block
// level in that block) dominates everything inside the target block.
static bool isAboveTargetBlock(Operation *def, Block *targetBlock,
                               DominanceInfo &domInfo) {
  return domInfo.properlyDominates(def->getBlock(), targetBlock);
}

// Recursively collect the closure of `producer` that has to be sunk into
// `targetBlock` together with the producer, grown to a fixpoint
static SmallVector<Operation *>
buildForwardClosure(Operation *producer, Block *targetBlock, Operation *&anchor,
                    const DenseMap<Operation *, unsigned> &pos,
                    CVPipeline::SyncWall &wall, DominanceInfo &domInfo) {
  llvm::SetVector<Operation *> cluster;
  cluster.insert(producer);

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *cur : llvm::to_vector(cluster)) {
      // Forward: uses of the current op.
      for (Operation *user : cur->getUsers()) {
        Operation *userAnc = CVPipeline::getAncestorInBlock(user, targetBlock);
        if (userAnc) {
          // Use already lives in the target block; track the earliest one.
          if ((!anchor || userAnc->isBeforeInBlock(anchor)) &&
              wall.hasSyncBetween(cur, userAnc)) {
            anchor = userAnc;
          }
          if (wall.hasSyncBetween(cur, anchor)) {
            changed |= cluster.insert(user);
          }
          continue;
        }
        // A use outside the target block must be pulled down recursively.
        if (user->hasTrait<OpTrait::IsTerminator>()) {
          return {}; // cannot sink past a terminator (e.g. loop-carried yield)
        }
      }

      // Backward: producers of the current op's (effective) operands.
      for (Value operand : cur->getOperands()) {
        if (!isa<MemRefType>(operand.getType())) {
          LDBG("Not a memref tyoe for operand: " << operand);
          continue;
        }
        Operation *def = operand.getDefiningOp();
        if (!def || cluster.contains(def)) {
          continue; // block argument or already relocated
        }
        LDBG("collect def op: " << *def);
        if (domInfo.properlyDominates(def->getBlock(), targetBlock)) {
          LDBG("Defining op already dominates the target block");
          continue;
        }
        if (def->hasTrait<OpTrait::IsTerminator>()) {
          return {};
        }
        // Pull the operand producer only when every use of it travels with the
        // cluster, otherwise moving it would break its outside users.
        if (cluster.insert(def)) {
          changed = true;
          for (Operation *user : def->getUsers()) {
            if (!cluster.contains(user)) {
              cluster.insert(def);
            }
          }
        }
      }
    }
  }

  if (!anchor) {
    return {};
  }

  SmallVector<Operation *> ordered(cluster.begin(), cluster.end());
  llvm::stable_sort(ordered, [&](Operation *a, Operation *b) {
    return pos.lookup(a) < pos.lookup(b);
  });
  return ordered;
}

// A memref producer whose use crosses a sync op, together with the whole
// forward closure that has to be sunk into the consumer's block.
struct Relocation {
  Operation *anchor; // earliest in-targetBlock use of the closure
  SmallVector<Operation *> cluster; // closure members in original pre-order
  int targetId;
  Attribute coreType;
};

// Relocate memref producers whose use crosses a sync op into the consumer's
// block. Every op is scanned for memref-typed operands: when the operand's
// producer op and the consumer are separated by a sync, the producer and its
// whole forward closure (uses, and the uses of those uses' results, ...) are
// re-partitioned to the consumer's group and moved to just before the earliest
// cross-sync use in the consumer's block. The closure is sunk only when every
// use ends up in the consumer's block on the far side of the sync, so the move
// preserves SSA dominance and the re-partition cannot straddle the barrier.
static void relocateMemrefDecls(ModuleOp module) {
  CVPipeline::ComputeBlockIdManager bm(module);
  DenseMap<Block *, CVPipeline::SyncWall> walls;
  DenseMap<Operation *, unsigned> pos;
  unsigned idx = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) { pos[op] = idx++; });

  DominanceInfo domInfo(module);
  SmallVector<Relocation> worklist;

  module.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!isa<MemRefType>(operand.getType())) {
        continue;
      }
      Operation *producer = operand.getDefiningOp();
      if (!producer) {
        continue; // block argument or already scheduled for relocation
      }

      // Determine the block the two ops share and check that a sync op sits
      // strictly between the producer and the consumer.
      Block *commonBlock = producer->getBlock();
      Operation *pAnchor = producer;
      Operation *cAnchor = nullptr;
      if (producer->getBlock() == op->getBlock()) {
        cAnchor = op;
      } else if (Operation *cAnc =
                     CVPipeline::getAncestorInBlock(op, producer->getBlock())) {
        // Consumer lives inside a region nested under the producer's block.
        cAnchor = cAnc;
      } else {
        continue; // producer is not in the consumer's block hierarchy
      }

      CVPipeline::SyncWall &wall =
          walls.try_emplace(commonBlock, commonBlock).first->second;
      if (!wall.hasSyncBetween(pAnchor, cAnchor)) {
        continue; // no sync in between -> nothing to relocate across
      }

      Block *targetBlock = op->getBlock();
      int targetId = bm.getBlockIdByOp(op);
      if (targetId == -1) {
        continue; // consumer not assigned to a group yet
      }

      // Pull the producer and every use that does not live in the target block
      // (recursively, including the uses of those uses' results) into the
      // target block.
      Operation *anchor = cAnchor;
      SmallVector<Operation *> ordered = buildForwardClosure(
          producer, targetBlock, anchor, pos, wall, domInfo);
      if (ordered.empty()) {
        LDBG("Failed to build closure for op: " << *op);
        continue;
      }
      // The earliest far-side use must sit past the sync, otherwise a use
      // above the barrier keeps the producer above it.
      if (commonBlock == targetBlock &&
          !wall.hasSyncBetween(producer, anchor)) {
        continue;
      }
      // Every member's operands must still dominate the insertion point.
      llvm::SetVector<Operation *> cluster(ordered.begin(), ordered.end());
      if (!allOperandsDominate(cluster, anchor, domInfo)) {
        continue;
      }

      worklist.push_back(
          {anchor, ordered, targetId, op->getAttr(CVPipeline::kCoreType)});
    }
  });

  for (const Relocation &reloc : worklist) {
    auto insertPt = reloc.anchor;
    for (auto it = reloc.cluster.rbegin(); it != reloc.cluster.rend(); ++it) {
      auto member = *it;
      // Move the member to just before the earliest cross-sync use (aka the
      // anchor) then re-partition it to the consumer's group (block id and core
      // type).
      member->moveBefore(insertPt);
      insertPt = member;
      bm.updateBlockId(member, reloc.targetId);

      LDBG("moved memref producer " << *member << " to block_id "
                                    << reloc.targetId);
    }
  }
}

} // namespace

class RelocateMemrefDeclPass
    : public PassWrapper<RelocateMemrefDeclPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelocateMemrefDeclPass)

  RelocateMemrefDeclPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "relocate-memref-decl"; }
  llvm::StringRef getDescription() const final {
    return "Relocate memref declarations whose uses cross a sync op into the "
           "consumer block and pull the dependency cluster with them";
  }
};

void RelocateMemrefDeclPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  // Without any sync op there is nothing a memref dependency could cross, so
  // the pass has no work to do.
  bool hasSync = false;
  module.walk([&](Operation *op) {
    if (CVPipeline::isSyncOp(op)) {
      hasSync = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasSync) {
    return;
  }

  LDBG("Before :\n" << *module);

  // relocate memref declarations whose uses cross a sync op into the consumer
  // block and pull their dependency cluster with them, so every declaration
  // ends up on the same side of each sync as its uses before SplitDataflow
  // lowers the barriers.
  relocateMemrefDecls(module);

  LDBG("After :\n" << *module);
}

std::unique_ptr<OperationPass<ModuleOp>> createRelocateMemrefDeclPass() {
  return std::make_unique<RelocateMemrefDeclPass>();
}

} // namespace triton
} // namespace mlir
