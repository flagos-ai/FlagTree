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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "broadcast-ub-opt";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;

namespace mlir {
namespace triton {

class BroadcastUBOptPass
    : public PassWrapper<BroadcastUBOptPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastUBOptPass)

  BroadcastUBOptPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "broadcast-ub-opt"; }
  llvm::StringRef getDescription() const final {
    return "Optimize broadcast by moving it to the block of its users when "
           "safe";
  }
};

} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {

void BroadcastUBOptPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(moduleOp, aa);
  CVPipeline::ComputeBlockIdManager bm(moduleOp);
  LOG_DEBUG(moduleOp);
  moduleOp.walk([&](linalg::BroadcastOp op) {
    if (CVPipeline::getOpCoreType(op.getOperation()) !=
        CVPipeline::CoreType::VECTOR_ONLY)
      return;
    if (op->getUsers().empty())
      return;

    // Just Simple scenario:
    // brc's user all in same Block and in same block id.
    Operation *oneUser = *op->getUsers().begin();
    if (oneUser->getBlock() != op->getBlock()) {
      return;
    }
    int firstUserBlockId = bm.getBlockIdByOp(oneUser);
    if (firstUserBlockId == -1 || CVPipeline::getOpCoreType(oneUser) !=
                                      CVPipeline::CoreType::VECTOR_ONLY) {
      // No blcok Id (means control flow) && not vector: skip;
      // vector only control flow should skip too.
      return;
    }
    bool allUsersSameBlock = llvm::all_of(op->getUsers(), [&](Operation *user) {
      return bm.getBlockIdByOp(user) == firstUserBlockId;
    });
    if (!allUsersSameBlock)
      return;
    LOG_DEBUG("broadcast " << *op << " \n all users in same block "
                           << firstUserBlockId << "\n");

    int broadcastBlockId = bm.getBlockIdByOp(op.getOperation());
    if (broadcastBlockId == firstUserBlockId) {
      LOG_DEBUG("broadcast already in same block, skip\n");
      return;
    }

    SmallVector<Operation *> opsToCheck = {op};
    if (CVPipeline::willCreateCycle(opsToCheck, memGraph, firstUserBlockId,
                                    bm)) {
      LOG_DEBUG("would create cycle, skip\n");
      return;
    }
    LOG_DEBUG("moving broadcast to block " << firstUserBlockId << "\n");
    bm.updateBlockId(op, firstUserBlockId);
  });
}

std::unique_ptr<OperationPass<ModuleOp>> createBroadcastUBOptPass() {
  return std::make_unique<BroadcastUBOptPass>();
}

} // namespace triton
} // namespace mlir
