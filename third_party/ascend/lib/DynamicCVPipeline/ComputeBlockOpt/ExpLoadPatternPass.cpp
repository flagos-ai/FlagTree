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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "extract-exp-load-pattern";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct ExpLoadPattern {
  linalg::BroadcastOp broadcastOp;
  Operation *expOp = nullptr;
  bufferization::ToTensorOp toTensorOp;
  memref::CopyOp copyOp;
  memref::AllocOp allocOp;
  arith::MulFOp mulfOp;
  int blockId = -1;
  int newBlockId = -1;
};

static bool isExpLike(Operation *op) {
  return isa_and_present<math::ExpOp, math::Exp2Op>(op);
}

// Match pattern:
//   mulf <- broadcast <- exp/exp2 <- load
//        ↖
//           %arg
static bool matchExpLoadPattern(linalg::BroadcastOp broadcastOp,
                                CVPipeline::ComputeBlockIdManager &bm,
                                ExpLoadPattern &info) {
  if (CVPipeline::getOpCoreType(broadcastOp) !=
      CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }
  int blockId = bm.getBlockIdByOp(broadcastOp);
  if (blockId == -1) {
    return false;
  }

  if (broadcastOp.getDpsInputs().empty()) {
    return false;
  }
  Operation *expOp = broadcastOp.getDpsInputs()[0].getDefiningOp();
  if (!isExpLike(expOp) || bm.getBlockIdByOp(expOp) != blockId ||
      !expOp->hasOneUse()) {
    return false;
  }

  auto toTensorOp =
      expOp->getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp || bm.getBlockIdByOp(toTensorOp) != blockId ||
      !toTensorOp->hasOneUse()) {
    return false;
  }

  auto allocOp = toTensorOp.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!allocOp || bm.getBlockIdByOp(allocOp) != blockId) {
    return false;
  }

  memref::CopyOp copyOp;
  for (Operation *user : allocOp->getUsers()) {
    auto c = dyn_cast<memref::CopyOp>(user);
    if (c && c.getTarget() == allocOp.getMemref()) {
      copyOp = c;
      break;
    }
  }
  if (!copyOp || bm.getBlockIdByOp(copyOp) != blockId ||
      copyOp->getBlock() != broadcastOp->getBlock()) {
    return false;
  }

  if (!broadcastOp->hasOneUse()) {
    return false;
  }
  auto mulfOp = dyn_cast<arith::MulFOp>(*broadcastOp->getUsers().begin());
  if (!mulfOp) {
    return false;
  }

  int newBlockId = bm.getBlockIdByOp(mulfOp);
  if (newBlockId == blockId) {
    return false;
  }
  if (CVPipeline::getOpCoreType(mulfOp) != CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }

  Value broadcastResult = broadcastOp->getResult(0);
  Value otherOperand =
      (mulfOp.getLhs() == broadcastResult) ? mulfOp.getRhs() : mulfOp.getLhs();
  if (!isa<BlockArgument>(otherOperand)) {
    return false;
  }

  info.broadcastOp = broadcastOp;
  info.expOp = expOp;
  info.toTensorOp = toTensorOp;
  info.copyOp = copyOp;
  info.allocOp = allocOp;
  info.mulfOp = mulfOp;
  info.blockId = blockId;
  info.newBlockId = newBlockId;
  return true;
}

} // namespace

class ExpLoadPatternPass
    : public PassWrapper<ExpLoadPatternPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpLoadPatternPass)

  ExpLoadPatternPass() = default;

  StringRef getArgument() const override { return "exp-load-pattern"; }

  StringRef getDescription() const override {
    return "Match load->exp->broadcast from a VECTOR block into a fresh "
           "VECTOR block";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    SmallVector<ExpLoadPattern> patterns;
    module.walk([&](linalg::BroadcastOp broadcastOp) {
      ExpLoadPattern info;
      if (matchExpLoadPattern(broadcastOp, bm, info)) {
        LOG_DEBUG("Matched exp-load pattern in block " << info.blockId);
        patterns.push_back(info);
      }
    });

    for (auto &info : patterns) {
      if (bm.getBlockIdByOp(info.broadcastOp) != info.blockId) {
        LOG_DEBUG("Stale pattern (block id changed), skip");
        continue;
      }

      SmallVector<Operation *> coreOps = {info.broadcastOp, info.expOp,
                                          info.toTensorOp, info.copyOp,
                                          info.allocOp};

      CVPipeline::SplitIf::ScalarClosure closure{info.broadcastOp->getBlock(),
                                                 coreOps, false};
      closure.collect();

      llvm::SetVector<Operation *> matchedOps;
      for (auto *op : coreOps) {
        matchedOps.insert(op);
      }
      for (Operation *op : closure.scalarOps) {
        matchedOps.insert(op);
      }
      CVPipeline::cloneScalarOpsForCrossBlockUses(bm, matchedOps,
                                                  info.newBlockId);

      if (!CVPipeline::willCreateCycle(matchedOps.getArrayRef(), memGraph,
                                       info.newBlockId, bm)) {
        for (Operation *op : matchedOps) {
          bm.updateBlockIdWithInner(op, info.newBlockId);
        }

        LOG_DEBUG("Process exp-load pattern: block " << info.blockId << " -> "
                                                     << info.newBlockId);
      }
    }
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpLoadPatternPass() {
  return std::make_unique<ExpLoadPatternPass>();
}

} // namespace triton
} // namespace mlir
