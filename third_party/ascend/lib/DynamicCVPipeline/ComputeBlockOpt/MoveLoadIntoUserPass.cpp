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
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

static constexpr const char *DEBUG_TYPE = "move-load-into-user";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

class MoveLoadIntoUserPass
    : public PassWrapper<MoveLoadIntoUserPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MoveLoadIntoUserPass)

  MoveLoadIntoUserPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "move-load-into-user"; }

private:
  struct LoadPatternInfo {
    SmallVector<Operation *> matchedOps;
    int commonBlockId;
    bufferization::ToTensorOp toTensorOp;
    memref::CopyOp copyOp;
    memref::AllocOp allocOp; // destination
  };

  bool matchLoadPattern(LoadPatternInfo &info);

  bool checkAllOpsInSameBlock(LoadPatternInfo &info,
                              CVPipeline::ComputeBlockIdManager &bm);
};

static Operation *traceCalculateUser(Operation *originUser,
                                     CVPipeline::ComputeBlockIdManager &bm,
                                     int totensorId,
                                     SmallVector<Operation *> &userChain) {
  // only find single chian
  //  pattern -> collapse1(totensorId)->collapse2(totensorId)...->user
  //  (targetblockId), collapse should move together.
  //         \-> other userChain   -> user (targetblockId)
  auto retUser = originUser;
  while (bm.getBlockIdByOp(retUser) == totensorId &&
         isa<tensor::CollapseShapeOp>(retUser)) {
    if (retUser->getResult(0).getNumUses() != 1) {
      break;
    }
    userChain.push_back(retUser);
    retUser = *retUser->getUsers().begin();
  }
  return retUser;
}

static std::optional<Operation *>
getFirstUser(bufferization::ToTensorOp toTensorOp,
             CVPipeline::ComputeBlockIdManager &bm,
             SmallVector<Operation *> &matchedOps) {
  Operation *firstUser = nullptr;
  SmallVector<Operation *> fistUserChain;

  for (auto *user : toTensorOp->getUsers()) {
    auto *userInBlock =
        CVPipeline::getAncestorInBlock(user, toTensorOp->getBlock());
    if (userInBlock) {
      SmallVector<Operation *> userChain;
      auto calUser = traceCalculateUser(
          userInBlock, bm, bm.getBlockIdByOp(toTensorOp), userChain);
      if (!firstUser || calUser->isBeforeInBlock(firstUser)) {
        // First in ordered IR, it's cblock is first too.
        firstUser = calUser;
        fistUserChain = userChain;
      }
    }
  }

  if (!firstUser) {
    LOG_DEBUG("No users of to_tensor found");
    return std::nullopt;
  }
  LOG_DEBUG("Find first user:\n" << *firstUser);

  if (CVPipeline::getOpCoreType(firstUser) != CVPipeline::VECTOR_ONLY) {
    LOG_DEBUG("First User is not VECTOR");
    return std::nullopt;
  }

  int blockId = bm.getBlockIdByOp(firstUser);
  if (blockId == -1) {
    LOG_DEBUG("First user has no block_id");
    return std::nullopt;
  }

  if (blockId == bm.getBlockIdByOp(toTensorOp)) {
    LOG_DEBUG("First user in same block");
    return std::nullopt;
  }
  matchedOps.append(fistUserChain.begin(), fistUserChain.end());
  return firstUser;
}

bool MoveLoadIntoUserPass::checkAllOpsInSameBlock(
    LoadPatternInfo &info, CVPipeline::ComputeBlockIdManager &bm) {
  if (info.matchedOps.empty()) {
    return false;
  }

  int commonBlockId = -1;
  for (auto *op : info.matchedOps) {
    int blockId = bm.getBlockIdByOp(op);
    if (blockId == -1) {
      LOG_DEBUG("Op has no block_id");
      return false;
    }
    if (commonBlockId == -1) {
      commonBlockId = blockId;
    } else if (blockId != commonBlockId) {
      LOG_DEBUG("Not all ops in the same block");
      return false;
    }
  }
  info.commonBlockId = commonBlockId;
  return true;
}

memref::AllocOp getOriginAlloc(Value dest) {
  if (auto allocOp = dest.getDefiningOp<memref::AllocOp>()) {
    return allocOp;
  } else if (auto viewLike =
                 dyn_cast<ViewLikeOpInterface>(dest.getDefiningOp())) {
    auto source = viewLike.getViewSource();
    if (source) {
      return getOriginAlloc(source);
    }
  }
  return nullptr;
}

bool MoveLoadIntoUserPass::matchLoadPattern(LoadPatternInfo &info) {
  // Step 1: Check if source is from global memory
  SetVector<Operation *> sourceViewOps;
  if (!CVPipeline::collectViewOpsAndCheckGlobalMemory(info.copyOp.getSource(),
                                                      sourceViewOps)) {
    LOG_DEBUG("Copy source is not from global memory");
    return false;
  }

  // Step 2: Find the pattern: reinterpret_cast/alloc -> memref.copy ->
  // to_tensor Get the reinterpret_cast (or alloc)
  Value dest = info.copyOp.getTarget();

  // Check if dest is alloc/from one alloc
  if (auto allocOp = getOriginAlloc(dest)) {
    info.allocOp = allocOp;
  } else {
    LOG_DEBUG("Copy dest is not from alloc");
    return false;
  }

  // Step 3: Find the to_tensor op
  for (auto *user : info.allocOp->getUsers()) {
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
    if (toTensor) {
      info.toTensorOp = toTensor;
      break;
    }
  }

  if (!info.toTensorOp) {
    LOG_DEBUG("No to_tensor op found after copy");
    return false;
  }

  // Collect matched ops: destOp, copyOp, toTensor
  info.matchedOps.push_back(info.allocOp);
  info.matchedOps.push_back(info.copyOp);
  info.matchedOps.push_back(info.toTensorOp);

  return true;
}

static void
collectAllDependencies(Operation *op, SetVector<Operation *> &deps,
                       int commonBlockId, CVPipeline::ComputeBlockIdManager &bm,
                       CVPipeline::MemoryDependenceGraph &memGraph) {
  if (deps.contains(op)) {
    return;
  }

  int blockId = bm.getBlockIdByOp(op);
  if (blockId == -1 || blockId != commonBlockId) {
    return;
  }

  deps.insert(op);

  CVPipeline::DependencyHelper depHelper(memGraph);
  depHelper.forEachSource(op, [&](Operation *source) {
    collectAllDependencies(source, deps, commonBlockId, bm, memGraph);
    return WalkResult::advance();
  });
}

void MoveLoadIntoUserPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(module, aliasAnalysis);
  auto bm = CVPipeline::ComputeBlockIdManager(module);

  SmallVector<LoadPatternInfo> validPatterns;

  LOG_DEBUG("before MogeLoadIntoUserPass ....\n" << *module);

  module.walk([&](memref::CopyOp copyOp) {
    LOG_DEBUG("\nNow check memref.copy = " << copyOp << "\n");
    // Step 1: Match the load pattern and collect 4 ops
    LoadPatternInfo info;
    info.copyOp = copyOp;
    if (!matchLoadPattern(info)) {
      return;
    }

    // Step 2: Check if all 3 ops are in the same block
    if (!checkAllOpsInSameBlock(info, bm)) {
      return;
    }
    // Store the valid pattern
    validPatterns.push_back(info);
  });

  // Process each valid pattern
  for (auto &pattern : validPatterns) {
    auto &matchedOps = pattern.matchedOps;
    int commonBlockId = pattern.commonBlockId;

    // Find first user of to_tensor
    auto firstUserOpt = getFirstUser(pattern.toTensorOp, bm, matchedOps);
    if (!firstUserOpt) {
      continue;
    }

    Operation *firstUser = firstUserOpt.value();
    int targetBlockId = bm.getBlockIdByOp(firstUser);

    SetVector<Operation *> opsToMove;
    for (auto *op : matchedOps) {
      collectAllDependencies(op, opsToMove, commonBlockId, bm, memGraph);
    }
    CVPipeline::cloneScalarOpsForCrossBlockUses(bm, opsToMove,
                                                bm.getBlockIdByOp(firstUser));
    // Check for cycles before moving
    if (CVPipeline::willCreateCycle(opsToMove.getArrayRef(), memGraph,
                                    targetBlockId, bm)) {
      LOG_DEBUG("Moving would create a cycle, skip(" << commonBlockId << ")");
      continue;
    }

    // Update block_id for all ops
    for (auto *op : opsToMove) {
      bm.updateBlockIdWithInner(op, targetBlockId);
    }

    LOG_DEBUG("Successfully moved ops to block " << targetBlockId);
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createMoveLoadIntoUserPass() {
  return std::make_unique<MoveLoadIntoUserPass>();
}

} // namespace triton
} // namespace mlir
