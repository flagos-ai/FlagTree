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

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-vector-if-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

/**
 * @brief Check whether an scf.if only contains Vector-tagged operations
 *
 * A pure-vector if has every (non-terminator) operation inside its regions
 * tagged as a VECTOR core op. Any operation that holds nested regions
 * (i.e. nested control flow such as scf.if / scf.for) is rejected because
 * such control ops carry no block tag and therefore cannot be merged.
 */
static bool isPureVectorIf(scf::IfOp ifOp) {
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        // Nested control flow has no Vector tag -> illegal to merge.
        if (op.getNumRegions() > 0) {
          return false;
        }
        if (CVPipeline::getOpCoreType(&op) !=
            CVPipeline::CoreType::VECTOR_ONLY) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * @brief Check whether a group of ops (an existing block_id's ops) is safe to
 *        fold into a pure-VECTOR target block_id.
 *
 * block_id is consumed downstream both as a scheduling/reorder key
 * (ReorderOpsByBlockIdPass) and as a CUBE/VECTOR compute-block boundary
 * (DataDependencyAnalysis's collectBlockInfo). Merging a block that contains
 * any CUBE-tagged op into a VECTOR-only block_id would silently produce a
 * single block_id spanning both core types, which downstream passes do not
 * expect. Sync ops are assigned unique block ids and must not be merged here.
 * A group op may itself be a control-flow op (e.g. scf.if / scf.for) that
 * carries no direct core_type tag, so getCoreTypeOfSimpleOpOrCf is used to
 * recurse into its body instead of getOpCoreType.
 */
static bool isPureVectorOpGroup(ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    if (CVPipeline::isSyncOp(op)) {
      return false;
    }
    if (CVPipeline::getCoreTypeOfSimpleOpOrCf(op) !=
        CVPipeline::CoreType::VECTOR_ONLY) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Determine the upstream block_id to merge the scf.if into
 *
 * The upstream is the data source of the if: its condition plus every value
 * consumed inside the regions that is produced outside the if. Constant-like
 * producers and values produced inside the if are ignored. When all remaining
 * data sources share a single block_id, that id is the merge target.
 */
static LogicalResult getUpstreamBlockId(scf::IfOp ifOp,
                                        CVPipeline::ComputeBlockIdManager &bm,
                                        int &target) {
  Block *parent = ifOp->getBlock();
  llvm::SmallDenseSet<int, CVPipeline::INIT_SIZE> ids;

  auto addSource = [&](Value v) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      return; // block argument: not a tagged data source
    }
    if (def->hasTrait<OpTrait::ConstantLike>()) {
      return; // constants are ignored
    }
    if (ifOp->isAncestor(def)) {
      return; // produced inside the if itself, not external source
    }
    Operation *anc = CVPipeline::getAncestorInBlock(def, parent);
    if (!anc) {
      return; // produced in an outer scope, not a sibling block op
    }
    int bid = bm.getBlockIdByOp(anc);
    if (bid != -1) {
      ids.insert(bid);
    }
  };

  addSource(ifOp.getCondition());
  ifOp->walk([&](Operation *op) {
    if (op == ifOp.getOperation()) {
      return;
    }
    for (Value v : op->getOperands()) {
      addSource(v);
    }
  });

  if (ids.size() != 1) {
    LOG_DEBUG("[getUpstreamBlockId] data sources are not consistent, count="
              << ids.size());
    return failure();
  }
  target = *ids.begin();
  return success();
}

static void applyMerge(scf::IfOp ifOp, ArrayRef<Operation *> downstreamOps,
                       int target, CVPipeline::ComputeBlockIdManager &bm) {
  bm.updateBlockId(ifOp.getOperation(), target);
  // Rewrite the inner ops so the whole if body shares the upstream block_id.
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        // Never fold a sync into a compute block: it must keep its own unique
        // block id so the fence between before/after ops survives.
        if (CVPipeline::isSyncOp(&op)) {
          continue;
        }
        if (CVPipeline::getOpBlockId(&op)) {
          bm.updateBlockId(&op, target);
        }
      }
    }
  }
  for (Operation *op : downstreamOps) {
    bm.updateBlockId(op, target);
  }
}

/**
 * @brief Collect the downstream block_ids that consume the if results,
 *        ordered by their first appearance in the parent block.
 */
static SmallVector<int>
collectDownstreamBlockIds(scf::IfOp ifOp, int target,
                          CVPipeline::ComputeBlockIdManager &bm) {
  Block *parent = ifOp->getBlock();
  llvm::DenseSet<Operation *> userAnchors;
  for (Value res : ifOp->getResults()) {
    for (Operation *user : res.getUsers()) {
      if (auto *anc = CVPipeline::getAncestorInBlock(user, parent)) {
        userAnchors.insert(anc);
      }
    }
  }

  // Collect the users with the program order.
  SmallVector<int> ordered;
  llvm::SmallDenseSet<int, CVPipeline::INIT_SIZE> seen;
  for (Operation &op : *parent) {
    if (!userAnchors.contains(&op)) {
      continue;
    }
    int bid = bm.getBlockIdByOp(&op);
    if (bid == -1 || bid == target) {
      continue;
    }
    if (seen.insert(bid).second) {
      ordered.push_back(bid);
    }
  }
  return ordered;
}

/**
 * @brief Try to merge one pure-vector scf.if with its upstream and a downstream
 * block
 */
static void tryMergeIf(scf::IfOp ifOp,
                       const CVPipeline::MemoryDependenceGraph &memGraph,
                       CVPipeline::ComputeBlockIdManager &bm) {
  int target = -1;
  if (failed(getUpstreamBlockId(ifOp, bm, target))) {
    return;
  }
  LOG_DEBUG("[tryMergeIf] upstream target block_id = " << target << " for "
                                                       << *ifOp);

  SmallVector<int> downstream = collectDownstreamBlockIds(ifOp, target, bm);

  // Prefer merging the if together with the first downstream block that keeps
  // the dependency graph acyclic, forming one large block.
  for (int bid : downstream) {
    SmallVector<Operation *> downstreamOps = bm.getOpsByBlockId(bid);
    if (!isPureVectorOpGroup(downstreamOps)) {
      LOG_DEBUG("[tryMergeIf] downstream block_id "
                << bid << " contains non-VECTOR ops, skip to avoid mixed "
                << "core_type block_id");
      continue;
    }
    SmallVector<Operation *> opsToUnify;
    opsToUnify.push_back(ifOp.getOperation());
    opsToUnify.append(downstreamOps.begin(), downstreamOps.end());
    if (!CVPipeline::willCreateCycle(opsToUnify, memGraph, target, bm)) {
      LOG_DEBUG("[tryMergeIf] merging if with downstream block_id "
                << bid << " into " << target);
      applyMerge(ifOp, downstreamOps, target, bm);
      return;
    }
    LOG_DEBUG("[tryMergeIf] downstream block_id "
              << bid << " would create a cycle, skip");
  }

  // Fallback: at least fold the if into the upstream block if that is safe.
  SmallVector<Operation *> ifOnly = {ifOp.getOperation()};
  if (!CVPipeline::willCreateCycle(ifOnly, memGraph, target, bm)) {
    LOG_DEBUG("[tryMergeIf] merging if into upstream "
              << target << " without downstream");
    applyMerge(ifOp, {}, target, bm);
  }
}

} // anonymous namespace

class MergeVectorIfBlockPass
    : public PassWrapper<MergeVectorIfBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeVectorIfBlockPass)

  MergeVectorIfBlockPass() = default;

  StringRef getArgument() const override { return "merge-vector-if-block"; }

  StringRef getDescription() const override {
    return "Merge pure-Vector scf.if blocks with their upstream data-source "
           "block and a downstream consumer block";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    llvm::SmallVector<scf::IfOp> PureVectorIfOps;
    module.walk([&](scf::IfOp ifOp) {
      if (isPureVectorIf(ifOp)) {
        PureVectorIfOps.push_back(ifOp);
      }
    });

    for (scf::IfOp ifOp : PureVectorIfOps) {
      tryMergeIf(ifOp, memGraph, bm);
    }

    LOG_DEBUG("After: " << *module);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeVectorIfBlockPass() {
  return std::make_unique<MergeVectorIfBlockPass>();
}

void registerMergeVectorIfBlockPass() {
  PassRegistration<MergeVectorIfBlockPass> reg;
}

} // namespace triton
} // namespace mlir
