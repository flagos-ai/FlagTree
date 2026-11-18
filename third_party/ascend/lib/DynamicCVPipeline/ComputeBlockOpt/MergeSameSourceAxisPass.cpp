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
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-same-source-axis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace triton {

namespace {

static bool isInSameRegion(Operation *op, Operation *source) {
  return op->getParentRegion() == source->getParentRegion();
}

// Walk parent chain from start to target (inclusive), deduped via inChain.
// Caller ensures target is on start's parent chain.
static void appendPath(Operation *start, Operation *target,
                       const llvm::DenseMap<Operation *, Operation *> &parent,
                       llvm::DenseSet<Operation *> &inChain,
                       SmallVectorImpl<Operation *> &chainOps) {
  for (Operation *pathOp = start; pathOp; pathOp = parent.lookup(pathOp)) {
    if (inChain.insert(pathOp).second) {
      chainOps.push_back(pathOp);
    }
    if (pathOp == target) {
      break;
    }
  }
}

// Append convergenceOp's same-block VECTOR_ONLY direct downstream so the moved
// op doesn't leave its tail behind.
static void
extendChainWithConvergenceTail(Operation *convergenceOp, Operation *source,
                               llvm::DenseSet<Operation *> &inChain,
                               SmallVectorImpl<Operation *> &chainOps) {
  auto convBlockIdOpt = CVPipeline::getOpBlockId(convergenceOp);
  if (!convBlockIdOpt || *convBlockIdOpt < 0) {
    return;
  }
  int convBlockId = *convBlockIdOpt;
  for (Operation *user : convergenceOp->getUsers()) {
    if (CVPipeline::getOpCoreType(user) != CVPipeline::CoreType::VECTOR_ONLY) {
      continue;
    }
    if (!isInSameRegion(user, source)) {
      continue;
    }
    auto userBidOpt = CVPipeline::getOpBlockId(user);
    if (!userBidOpt || *userBidOpt != convBlockId) {
      continue;
    }
    if (inChain.insert(user).second) {
      chainOps.push_back(user);
    }
  }
}

static bool findNearestConvergence(Operation *source,
                                   ArrayRef<Operation *> consumers,
                                   SmallVectorImpl<Operation *> &chainOps) {
  if (consumers.size() < 2) {
    return false;
  }

  llvm::DenseMap<Operation *, int> firstIdx;
  llvm::DenseMap<Operation *, Operation *> parent;
  SmallVector<std::pair<Operation *, int>> bfsQueue;

  for (size_t consumerIndex = 0; consumerIndex < consumers.size();
       ++consumerIndex) {
    Operation *consumer = consumers[consumerIndex];
    auto ins = firstIdx.insert({consumer, (int)consumerIndex});
    if (ins.second) {
      parent[consumer] = nullptr;
      bfsQueue.push_back({consumer, (int)consumerIndex});
    }
  }

  for (size_t queueIndex = 0; queueIndex < bfsQueue.size(); ++queueIndex) {
    Operation *cur = bfsQueue[queueIndex].first;
    int myIdx = bfsQueue[queueIndex].second;
    for (Operation *user : cur->getUsers()) {
      if (CVPipeline::getOpCoreType(user) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }
      if (!isInSameRegion(user, source)) {
        continue;
      }
      auto it = firstIdx.find(user);
      if (it == firstIdx.end()) {
        firstIdx[user] = myIdx;
        parent[user] = cur;
        bfsQueue.push_back({user, myIdx});
      } else if (it->second != myIdx) {
        if (llvm::is_contained(consumers, user)) {
          continue;
        }
        Operation *cons1 = consumers[it->second];
        Operation *cons2 = consumers[myIdx];
        Operation *convergenceOp = user;

        chainOps.clear();
        llvm::DenseSet<Operation *> inChain;
        inChain.clear();
        appendPath(convergenceOp, cons1, parent, inChain, chainOps);
        appendPath(cur, cons2, parent, inChain, chainOps);
        extendChainWithConvergenceTail(convergenceOp, source, inChain,
                                       chainOps);
        return true;
      }
    }
  }
  return false;
}

static void tryMergeSource(Operation *source,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           CVPipeline::ComputeBlockIdManager &bm) {
  auto srcBlockIdOpt = CVPipeline::getOpBlockId(source);
  if (!srcBlockIdOpt || *srcBlockIdOpt < 0) {
    return;
  }
  int srcBlockId = *srcBlockIdOpt;

  if (mlir::isa<mlir::arith::ConstantOp>(source)) {
    return;
  }

  // Source must be tensor-shaped; scalars have no axis semantics.
  if (source->getNumResults() != 1 ||
      !mlir::isa<mlir::TensorType>(source->getResult(0).getType())) {
    return;
  }

  SmallVector<Operation *> consumers;
  for (Operation *user : source->getUsers()) {
    if (CVPipeline::getOpCoreType(user) != CVPipeline::CoreType::VECTOR_ONLY) {
      continue;
    }
    if (!isInSameRegion(user, source)) {
      continue;
    }
    auto bidOpt = CVPipeline::getOpBlockId(user);
    if (!bidOpt || *bidOpt == srcBlockId) {
      continue;
    }
    consumers.push_back(user);
  }
  if (consumers.size() < 2) {
    return;
  }

  SmallVector<Operation *> chainOps;
  if (!findNearestConvergence(source, consumers, chainOps)) {
    return;
  }

  // Guard: convergenceOp's upstream operands must all live in blocks that this
  // merge will cover
  Operation *convergenceOp = chainOps.front();
  for (Value operand : convergenceOp->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    if (!isInSameRegion(defOp, source)) {
      continue;
    }
    if (defOp == source) {
      continue;
    }
    if (llvm::is_contained(chainOps, defOp)) {
      continue;
    }
    auto defBidOpt = CVPipeline::getOpBlockId(defOp);
    if (defBidOpt && *defBidOpt == srcBlockId) {
      continue;
    }
    LOG_DEBUG("[tryMergeSource] convergenceOp="
              << *convergenceOp << " has unaligned upstream defOp=" << *defOp
              << ", skip");
    return;
  }

  LOG_DEBUG("[tryMergeSource] candidate source=" << *source << " chainSize="
                                                 << chainOps.size());

  if (CVPipeline::willCreateCycle(chainOps, memGraph, srcBlockId, bm)) {
    LOG_DEBUG("[tryMergeSource] willCreateCycle, skip");
    return;
  }

  for (Operation *op : chainOps) {
    bm.updateBlockId(op, srcBlockId);
  }
  LOG_DEBUG("[tryMergeSource] merged " << chainOps.size()
                                       << " ops into block_id=" << srcBlockId);
}

} // anonymous namespace

class MergeSameSourceAxisPass
    : public PassWrapper<MergeSameSourceAxisPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeSameSourceAxisPass)

  MergeSameSourceAxisPass() = default;

  StringRef getArgument() const override { return "merge-same-source-axis"; }

  StringRef getDescription() const override {
    return "Rewrite ssbuffer.block_id so that >=2 vector consumers of a "
           "source op whose downstream converges at a common op end up in "
           "the same block as the source. Only VECTOR core ops are considered.";
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

    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (CVPipeline::getOpCoreType(op) != CVPipeline::CoreType::VECTOR_ONLY ||
          !CVPipeline::getOpBlockId(op) || op->getUsers().empty()) {
        return;
      }
      if (op->getNumResults() != 1 ||
          !mlir::isa<mlir::TensorType>(op->getResult(0).getType())) {
        return;
      }
      candidates.push_back(op);
    });

    for (Operation *source : candidates) {
      tryMergeSource(source, memGraph, bm);
    }

    LOG_DEBUG("After: " << *module);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMergeSameSourceAxisPass() {
  return std::make_unique<MergeSameSourceAxisPass>();
}

void registerMergeSameSourceAxisPass() {
  PassRegistration<MergeSameSourceAxisPass> reg;
}

} // namespace triton
} // namespace mlir
