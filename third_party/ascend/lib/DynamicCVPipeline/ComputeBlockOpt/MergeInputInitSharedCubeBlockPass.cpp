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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "merge-input-init-shared-cube-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

// Return the matmul whose result feeds both the consumer's init (directly) and
// one of its inputs (possibly through a trunc), or null if no such matmul.
static linalg::MatmulOp getSharedInputInitProducer(linalg::MatmulOp consumer) {
  auto inits = consumer.getDpsInits();
  if (inits.empty()) {
    return {};
  }
  auto initProducer =
      dyn_cast_if_present<linalg::MatmulOp>(inits.front().getDefiningOp());
  if (!initProducer) {
    return {};
  }
  for (Value input : consumer.getDpsInputs()) {
    auto inputProducer = dyn_cast_if_present<linalg::MatmulOp>(
        CVPipeline::getSourceThroughCIntermediateOps(input));
    if (inputProducer == initProducer) {
      return initProducer;
    }
  }
  return {};
}

// Merge the ops of consumerBlockId into producerBlockId when it does not
// create a cycle.
static LogicalResult
tryMergeBlocks(int consumerBlockId, int producerBlockId,
               const CVPipeline::MemoryDependenceGraph &memGraph,
               CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SmallVector<Operation *> consumerOps =
      bm.getOpsByBlockId(consumerBlockId);
  if (consumerOps.empty() ||
      CVPipeline::willCreateCycle(consumerOps, memGraph, producerBlockId, bm)) {
    return failure();
  }
  LOG_DEBUG("Merging block " << consumerBlockId << " into block "
                             << producerBlockId);
  for (Operation *op : consumerOps) {
    bm.updateBlockId(op, producerBlockId);
  }
  return success();
}

class MergeInputInitSharedCubeBlockPass
    : public PassWrapper<MergeInputInitSharedCubeBlockPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MergeInputInitSharedCubeBlockPass)

  MergeInputInitSharedCubeBlockPass() = default;

  StringRef getArgument() const override {
    return "merge-input-init-shared-cube-block";
  }

  StringRef getDescription() const override {
    return "Merge cube blocks where a matmul result is used as both input and "
           "init of the next matmul";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }
    LOG_DEBUG("Before: MergeInputInitSharedCubeBlockPass" << *module);

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    // Collect matmuls first; merging rewrites block_id in place.
    llvm::SmallVector<linalg::MatmulOp> matmuls;
    module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    for (linalg::MatmulOp consumer : matmuls) {
      auto producer = getSharedInputInitProducer(consumer);
      if (!producer) {
        continue;
      }

      auto producerBlockId = CVPipeline::getOpBlockId(producer);
      auto consumerBlockId = CVPipeline::getOpBlockId(consumer);
      if (!producerBlockId || !consumerBlockId ||
          *producerBlockId == *consumerBlockId) {
        continue;
      }

      if (failed(tryMergeBlocks(*consumerBlockId, *producerBlockId, memGraph,
                                bm))) {
        LOG_DEBUG("[error] matmul result in block_id = "
                  << *producerBlockId
                  << " is used by other ops and create cycle after merge");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createMergeInputInitSharedCubeBlockPass() {
  return std::make_unique<MergeInputInitSharedCubeBlockPass>();
}

} // namespace triton
} // namespace mlir
