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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"

static constexpr const char *DEBUG_TYPE = "unify-alloc-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct FillInfo {
  linalg::FillOp fillOp;
  scf::IfOp parentIf;
};

/**
 * @brief Collect direct users of alloc result
 *
 * This function collects all operations that directly use the alloc result,
 * excluding linalg.fill operations because linalg.fill uses BlockArgument
 * (i.e., the outs parameter) rather than SSA value dependency.
 *
 * @param allocResult The result value of memref.alloc
 * @return SmallVector<Operation*> List of direct user operations
 *
 * @note linalg.fill is a DestinationStyleOp where:
 *       - ins(%v : f16) is the input value
 *       - outs(%alloc : memref) is the target memory location (BlockArgument)
 *       Therefore, linalg.fill does not appear in allocResult.getUsers().
 */
static SmallVector<Operation *> collectDirectUsers(Value allocResult) {
  SmallVector<Operation *> directUsers;
  for (Operation *user : allocResult.getUsers()) {
    if (!isa<linalg::FillOp>(user)) {
      directUsers.push_back(user);
    }
  }
  return directUsers;
}

/**
 * @brief Get common block_id from a list of operations
 *
 * Checks whether all operations have the same block_id.
 * If they are the same, returns that block_id; otherwise returns std::nullopt.
 *
 * Special handling for memref.subview: if direct users include memref.subview,
 * we look through to memref.copy. If there's exactly one memref.copy reachable,
 * return its block_id (not the subview's). If multiple copies exist, return
 * error.
 *
 * @param ops List of operations to check
 * @return std::optional<int> Returns common block_id if all are the same,
 *         otherwise returns std::nullopt
 */
static LogicalResult getCommonBlockId(ArrayRef<Operation *> ops, int &blockId) {
  if (ops.empty()) {
    return failure();
  }

  llvm::SmallDenseSet<int, 4> copyBlockIds;
  for (Operation *op : ops) {
    if (isa<ViewLikeOpInterface>(op)) {
      for (auto *user : op->getUsers()) {
        if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
          if (auto blockId = CVPipeline::getOpBlockId(copyOp)) {
            copyBlockIds.insert(*blockId);
          }
        }
      }
    }
  }

  if (copyBlockIds.size() > 1) {
    LOG_DEBUG("[getCommonBlockId] Multiple block_ids found, ops: ");
    for (Operation *op : ops) {
      LOG_DEBUG("  " << *op);
    }
    return failure();
  }

  if (copyBlockIds.empty()) {
    LOG_DEBUG("[getCommonBlockId] There are not CopyOp!");
    for (Operation *op : ops) {
      LOG_DEBUG("  " << *op);
    }
    return failure();
  }

  blockId = *copyBlockIds.begin();
  return success();
}

/**
 * @brief Find linalg.fill operation that uses alloc as outs inside scf.if
 *
 * This function searches for linalg.fill operations that satisfy:
 * 1. Use the given alloc result as its outs parameter
 * 2. Located inside an scf.if operation (then branch only)
 * 3. The scf.if has no else region (withElseRegion=false)
 *
 * @param allocResult The alloc result value to search for
 * @return FillInfo Structure containing fillOp and parentIf if found
 */
static FillInfo findFillOpInSCFIf(Value allocResult) {
  FillInfo info;
  for (Operation *user : allocResult.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp) {
      continue;
    }

    auto parentIf = fillOp->getParentOfType<scf::IfOp>();
    if (!parentIf) {
      continue;
    }

    if (!parentIf.getElseRegion().empty()) {
      continue;
    }

    Block *parentBlock = fillOp->getBlock();
    if (parentBlock != &parentIf.getThenRegion().front()) {
      continue;
    }

    if (fillOp.getDpsInits()[0] == allocResult) {
      info.fillOp = fillOp;
      info.parentIf = parentIf;
      return info;
    }
  }
  return info;
}

/**
 * @brief Check if the scf.if containing linalg.fill has other operations
 *
 * Determines whether the then region of the scf.if that contains linalg.fill
 * also contains other operations besides linalg.fill. If it does, unification
 * must be aborted, because the linalg.fill is mixed with unrelated operations
 * and unifying the whole scf.if block_id would be incorrect.
 *
 * @param info FillInfo structure containing fillOp and parentIf
 * @return bool Returns true if there are other ops besides linalg.fill in the
 *         then region, false otherwise
 */
static bool hasOtherOpsInIf(const FillInfo &info) {
  if (!info.fillOp || !info.parentIf) {
    return false;
  }

  Block *fillBlock = info.fillOp->getBlock();
  int opCount = 0;
  for (auto &op : fillBlock->without_terminator()) {
    (void)op;
    opCount++;
  }
  return opCount > 1;
}

/// Trace back along the view-like chain from \p copyOp's source and collect
/// all ops. Only ops in the same Block as the copyOp are collected.
static void traceViewChain(memref::CopyOp copyOp,
                           SmallVectorImpl<Operation *> &result) {
  for (Value v = copyOp.getSource(); auto *defOp = v.getDefiningOp();) {
    if (defOp->getBlock() != copyOp->getBlock())
      break;
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(defOp))
      result.push_back(viewOp), v = viewOp.getViewSource();
    else if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp))
      result.push_back(sliceOp), v = sliceOp.getSource();
    else
      break;
  }
}

/**
 * @brief Collect source view chain ops from memref.copy's source operand
 *
 * For each memref.copy found by penetrating ViewLike direct users (subview of
 * alloc), trace back the copy's source operand along the ViewLikeOpInterface /
 * tensor.extract_slice chain (e.g., memref.subview -> memref.reinterpret_cast
 * -> tensor.extract_slice), and collect all ops in the chain.
 *
 * @param directUsers Direct users of alloc result (containing ViewLike ops
 *                    whose users include memref.copy)
 * @return SmallVector<Operation*> Collected source view chain ops
 *
 */
static SmallVector<Operation *>
collectSourceViewChainOps(ArrayRef<Operation *> directUsers) {
  SmallVector<Operation *> chainOps;
  for (Operation *op : directUsers) {
    if (!isa<ViewLikeOpInterface, tensor::ExtractSliceOp>(op)) {
      continue;
    }
    // Penetrate through view-like ops to find memref.copy users
    SmallVector<Operation *> worklist = {op};
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      for (auto *user : cur->getUsers()) {
        if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
          traceViewChain(copyOp, chainOps);
        } else if (isa<ViewLikeOpInterface, tensor::ExtractSliceOp>(user)) {
          worklist.push_back(user);
        }
      }
    }
  }
  return chainOps;
}

/**
 * @brief Try to unify block_id for a single alloc operation
 *
 * @param allocOp The memref.alloc operation to process
 * @param memGraph Memory dependence graph for cycle detection
 * @return LogicalResult Returns success if unification was performed, failure
 * otherwise
 */
static LogicalResult
tryUnifyForAlloc(memref::AllocOp allocOp,
                 const CVPipeline::MemoryDependenceGraph &memGraph,
                 CVPipeline::ComputeBlockIdManager &bm) {
  // Step1: Collect direct users (excluding linalg.fill)
  Value allocResult = allocOp.getResult();
  LOG_DEBUG("[tryUnifyForAlloc] start from allocOp: " << *allocOp);
  SmallVector<Operation *> directUsers = collectDirectUsers(allocResult);
  if (directUsers.empty()) {
    return success();
  }

  // Step2: Find linalg.fill inside scf.if that uses this alloc
  FillInfo fillInfo = findFillOpInSCFIf(allocResult);
  if (!fillInfo.fillOp) {
    return success();
  }
  LOG_DEBUG(
      "[tryUnifyForAlloc] Found fillOp in scf.if: " << *fillInfo.parentIf);

  // Step3: Check if all direct users have the same block_id
  int targetBlockId;
  if (failed(getCommonBlockId(directUsers, targetBlockId))) {
    LOG_DEBUG("allocOp has copyOp from different Block");
    return failure();
  }
  LOG_DEBUG("[getSameBlockId] GetSameBlockId: " << targetBlockId);

  // Step4: If the scf.if has other operations besides linalg.fill, abort
  if (hasOtherOpsInIf(fillInfo)) {
    LOG_DEBUG("[warning] SCF.IF has other ops, failed to unify");
    return success();
  }

  // Step5: collect allocOp, ifOp, fillOp, directUsers and ViewChainOps
  SmallVector<Operation *> coreOps = {
      allocOp.getOperation(),
      fillInfo.fillOp.getOperation(),
      fillInfo.parentIf.getOperation(),
  };
  coreOps.append(directUsers);
  coreOps.append(collectSourceViewChainOps(directUsers));

  // Step6: Cycle detection and block_id assignment
  if (CVPipeline::willCreateCycle(coreOps, memGraph, targetBlockId, bm)) {
    LOG_DEBUG("[error] Find cycle, have unsupport IR! Should Check!!");
    return failure();
  }
  for (auto *op : coreOps) {
    bm.updateBlockId(op, targetBlockId);
  }
  LOG_DEBUG("[tryUnifyForAlloc] Successfully unify block_id to "
            << targetBlockId << " for allocOp: " << *allocOp);
  return success();
}

} // anonymous namespace

class UnifyAllocBlockPass
    : public PassWrapper<UnifyAllocBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnifyAllocBlockPass)

  UnifyAllocBlockPass() = default;

  StringRef getArgument() const override { return "unify-alloc-block"; }

  StringRef getDescription() const override {
    return "Unify block_id for memref.alloc, scf.if with linalg.fill, and "
           "memref.subview operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    LOG_DEBUG("Before: " << *module << "\n");
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    llvm::SmallVector<memref::AllocOp> allocOps;

    module.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });

    for (memref::AllocOp allocOp : allocOps) {
      if (failed(tryUnifyForAlloc(allocOp, memGraph, bm))) {
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return;
      }
    }

    LOG_DEBUG("After: " << *module << "\n");
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnifyAllocBlockPass() {
  return std::make_unique<UnifyAllocBlockPass>();
}

void registerUnifyAllocBlockPass() {
  PassRegistration<UnifyAllocBlockPass> reg;
}

} // namespace triton
} // namespace mlir
