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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "unify-store-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

static bool isStoreOp(const Operation *op) {
  return isa<bufferization::MaterializeInDestinationOp>(op) ||
         isa<hivm::StoreOp>(op);
}

static Value getStoreSource(Operation *storeOp) {
  if (auto materialize =
          dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
    return materialize.getSource();
  }
  if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
    return hivmStore.getSrc();
  }
  return Value();
}

static Value getStoreDest(Operation *storeOp) {
  if (auto materialize =
          dyn_cast<bufferization::MaterializeInDestinationOp>(storeOp)) {
    return materialize.getDest();
  }
  if (auto hivmStore = dyn_cast<hivm::StoreOp>(storeOp)) {
    return hivmStore.getDst();
  }
  return Value();
}

static bool isViewOrExtractSliceOp(Operation *op) {
  return isa<ViewLikeOpInterface>(op) || isa<tensor::ExtractSliceOp>(op);
}

static Value getViewSourceValue(Operation *viewOp) {
  if (auto viewLike = dyn_cast<ViewLikeOpInterface>(viewOp)) {
    return viewLike.getViewSource();
  }
  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(viewOp)) {
    return extract.getSource();
  }
  return Value();
}

static Operation *traceProducerOp(Operation *storeOp,
                                  SmallVector<Operation *> &dataViewOps) {
  Value cur = getStoreSource(storeOp);
  while (Operation *defOp = cur.getDefiningOp()) {
    // view-like op (incl. tensor.extract_slice): transparent, pierce through
    if (isViewOrExtractSliceOp(defOp)) {
      dataViewOps.push_back(defOp);
      cur = getViewSourceValue(defOp);
      continue;
    }
    // scalar-producing op: skip
    if (CVPipeline::isScalarLike(cur)) {
      return nullptr; // producer is a scalar-like op
    }
    // hit: a real compute op; only unifiable if it has the same core type as
    // storeOp
    if (CVPipeline::getOpCoreType(defOp) !=
        CVPipeline::getOpCoreType(storeOp)) {
      LOG_DEBUG("Producer op is not VECTOR_ONLY: " << *defOp << "\n");
      return nullptr;
    }
    LOG_DEBUG("Find producer op: " << *defOp);
    return defOp;
  }
  LOG_DEBUG("Could not find producer op!");
  return nullptr; // reached a block argument (iter_arg / gm arg)
}

/**
 * @brief Collect view ops (data chain + dest chain) to unify.
 *
 * All view ops reachable from @p dataViewOps and the dest chain are appended
 * to @p ops. Duplicates are skipped via @p seen.
 */
static void collectViewOpsToUnify(SmallVector<Operation *> &ops,
                                  Operation *storeOp,
                                  const SmallVector<Operation *> &dataViewOps,
                                  SmallPtrSet<Operation *, 16> &seen) {
  auto addIfMatch = [&](Operation *op) {
    if (seen.insert(op).second && op->getBlock() == storeOp->getBlock()) {
      ops.push_back(op);
    }
  };

  for (Operation *viewOp : dataViewOps) {
    addIfMatch(viewOp);
  }

  // dest-chain view ops: walk up the dest memref's view chain.
  Value cur = getStoreDest(storeOp);
  while (Operation *defOp = cur.getDefiningOp()) {
    if (!isViewOrExtractSliceOp(defOp)) {
      break;
    }
    addIfMatch(defOp);
    cur = getViewSourceValue(defOp);
  }
}

/**
 * @brief Recursively collect scalar-like dependencies within the same big
 * block.
 *
 * For every scalar-like operand of each op in @p ops (and of @p storeOp's
 * control-flow ancestor ops, e.g. scf.for lb/ub/step), append its defining op
 * only if it lives in the same big block as @p storeOp. Scalars defined in
 * outer blocks (e.g. func-level constants) are skipped, because moving them
 * across blocks is unsafe and they are typically shared by many blocks.
 */
static void collectScalarDeps(SmallVector<Operation *> &ops, Operation *storeOp,
                              SmallPtrSet<Operation *, 16> &seen) {
  SmallVector<Operation *> worklist(ops.begin(), ops.end());

  for (Operation *p = storeOp->getParentOp(); p && !isa<ModuleOp>(p);
       p = p->getParentOp()) {
    if (isa<LoopLikeOpInterface>(p) || isa<scf::IfOp>(p)) {
      worklist.push_back(p);
    }
  }

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    for (Value operand : cur->getOperands()) {
      // only scalar-like operands are collected; non-scalar operands
      // belong to data/dest chains or external blocks
      if (!CVPipeline::isScalarLike(operand)) {
        continue;
      }
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        continue; // block argument, no block_id to change
      }
      // only collect scalar ops in the same big block as storeOp;
      // outer-block scalars (e.g. func-level constants) are skipped.
      if (defOp->getBlock() != storeOp->getBlock()) {
        continue;
      }
      if (!seen.insert(defOp).second) {
        continue;
      }
      ops.push_back(defOp);
      worklist.push_back(defOp); // recurse on its operands
    }
  }
}

/**
 * @brief Assemble the full set of ops whose block_id should be unified.
 *
 * opsToUnify = {store} + viewOps (data + dest) +
 *              scalarDeps (recursively collected).
 */
static SmallVector<Operation *>
collectOpsToUnify(Operation *storeOp,
                  const SmallVector<Operation *> &dataViewOps,
                  CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> opsToUnify;
  opsToUnify.push_back(storeOp);

  SmallPtrSet<Operation *, 16> seen;
  seen.insert(storeOp);

  collectViewOpsToUnify(opsToUnify, storeOp, dataViewOps, seen);
  collectScalarDeps(opsToUnify, storeOp, seen);

  for (Operation *op : opsToUnify) {
    LOG_DEBUG("opToUnify: " << *op);
  }
  return opsToUnify;
}

/**
 * @brief Match a store pattern: trace producer, collect ops to unify.
 *
 * @return true if the pattern is matched and ready for apply, false to skip.
 *
 * @note producer is inserted as the first element of @p matchedOps (required
 *       by cloneScalarOpsForCrossBlockUses to identify the target block),
 *       followed by the store op, view ops, and scalar dependencies.
 */
static bool matchStorePattern(Operation *storeOp,
                              CVPipeline::ComputeBlockIdManager &bm,
                              SetVector<Operation *> &matchedOps) {
  LOG_DEBUG("Start from storeOp: " << *storeOp);
  int storeBlockId = bm.getBlockIdByOp(storeOp);
  if (storeBlockId == -1) {
    LOG_DEBUG("storeOp has no block_id, cannot unify! ");
    return false;
  }

  // Step 1: trace producer op from store.source, skipping views + scalars,
  //         and collect all viewops in the data chain.
  SmallVector<Operation *> dataViewOps;
  Operation *producer = traceProducerOp(storeOp, dataViewOps);
  if (!producer) {
    return false; // block argument / fully-scalar chain / core_type mismatch ->
                  // skip
  }

  if (bm.getBlockIdByOp(producer) == -1 ||
      producer->getBlock() != storeOp->getBlock()) {
    LOG_DEBUG("ProducerOp has no block_id or not in the same block as storeOp, "
              "cannot unify! ");
    return false; // producer has no block_id, cannot unify
  }

  // Step 2: build matchedOps = {producer} + {store} + viewOps + scalarDeps.
  //         producer first is required by cloneScalarOpsForCrossBlockUses.
  matchedOps.insert(producer);
  SmallVector<Operation *> opsToUnify =
      collectOpsToUnify(storeOp, dataViewOps, bm);
  matchedOps.insert(opsToUnify.begin(), opsToUnify.end());
  return true;
}

/**
 * @brief Apply store unify: cycle detection + block_id update.
 *
 * @return true on success, false if cycle detected (pattern is skipped).
 */
static bool applyStoreUnify(const SetVector<Operation *> &matchedOps,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            CVPipeline::ComputeBlockIdManager &bm) {
  Operation *producer = matchedOps[0];
  int targetBlockId = bm.getBlockIdByOp(producer);

  SmallVector<Operation *> opsToUnifyList(matchedOps.begin(), matchedOps.end());
  if (CVPipeline::willCreateCycle(opsToUnifyList, memGraph, targetBlockId,
                                  bm)) {
    LOG_DEBUG("Cycle detected, cannot unify storeOp to producer!\n");
    return false;
  }

  for (Operation *op : matchedOps) {
    bm.updateBlockId(op, targetBlockId);
  }
  LOG_DEBUG("Successfully unify storeOps\n");
  return true;
}

} // anonymous namespace

class UnifyStoreBlockPass
    : public PassWrapper<UnifyStoreBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnifyStoreBlockPass)

  UnifyStoreBlockPass() = default;

  StringRef getArgument() const override { return "unify-store-block"; }

  StringRef getDescription() const override {
    return "Merge store-semantic operations into the producer vector compute "
           "block";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LOG_DEBUG("Before UnifyStoreBlock: " << *module);

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    // Phase 1: collect all matched store patterns (read-only, no modification).
    //          Only VECTOR-core stores are candidates.
    SmallVector<Operation *> storeOps;
    module.walk([&](Operation *op) {
      if (isStoreOp(op) &&
          CVPipeline::getOpCoreType(op) == CVPipeline::CoreType::VECTOR_ONLY) {
        storeOps.push_back(op);
      }
    });

    for (auto op : storeOps) {
      SetVector<Operation *> matchedOps;
      // Phase 1: collect all matched store patterns (read-only, no
      // modification).
      //          Only VECTOR-core stores are candidates.
      if (matchStorePattern(op, bm, matchedOps)) {
        // Phase 2: clone scalar ops shared between a pattern and other blocks,
        //          so that moving pattern ops into the producer's block_id does
        //          not create cross-block dependency cycles.
        //          In order to avoid cycle, clone scalar-like ops.
        //          A   ->   D
        //           ↘      ↗
        //            B -> C
        //          Now we want to fuse D to A, so clone C' scalarOps for
        //          D dependencies to avoid cycle.
        LOG_DEBUG("Producer op = " << *matchedOps[0]);
        CVPipeline::cloneScalarOpsForCrossBlockUses(
            bm, matchedOps, bm.getBlockIdByOp(matchedOps[0]));
        // Phase 3: for each pattern, cycle detection + block_id update.
        //          Rebuild bm because cloning may have added new ops.
        if (!applyStoreUnify(matchedOps, memGraph, bm)) {
          for (Operation *op : matchedOps) {
            LOG_DEBUG("Cannot set block id for ops: " << *op);
          }
        }
      }
    }

    LOG_DEBUG("After UnifyStoreBlock: " << *module);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnifyStoreBlockPass() {
  return std::make_unique<UnifyStoreBlockPass>();
}

void registerUnifyStoreBlockPass() {
  PassRegistration<UnifyStoreBlockPass> reg;
}

} // namespace triton
} // namespace mlir
