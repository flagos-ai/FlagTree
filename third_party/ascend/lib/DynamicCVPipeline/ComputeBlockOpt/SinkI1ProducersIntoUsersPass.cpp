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

#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

static constexpr const char *DEBUG_TYPE = "sink-i1-producers-into-users";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;

namespace mlir {
namespace triton {

class SinkI1ProducersIntoUsersPass
    : public PassWrapper<SinkI1ProducersIntoUsersPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkI1ProducersIntoUsersPass)

  SinkI1ProducersIntoUsersPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "sink-i1-producers-into-users";
  }
  llvm::StringRef getDescription() const final {
    return "Sink i1-producing ops next to each of their i1 uses";
  }
};

} // namespace triton
} // namespace mlir

namespace {

static bool isValidI1Producer(Operation *op) {
  if (isa<scf::SCFDialect>(op->getDialect())) {
    return false;
  }
  if (op->getNumResults() > 1) {
    return false;
  }
  for (auto result : op->getResults()) {
    if (auto tensorType = dyn_cast<mlir::TensorType>(result.getType())) {
      mlir::Type elemType = tensorType.getElementType();
      if (elemType.isInteger(1)) {
        return true;
      }
    }
  }
  return false;
}

static bool isPureAndRegionless(Operation *op) {
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    if (!effects.empty())
      return false;
  }
  return true;
}

} // namespace

namespace mlir {
namespace triton {

void SinkI1ProducersIntoUsersPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  CVPipeline::ComputeBlockIdManager bm(moduleOp);
  LOG_DEBUG(moduleOp);

  SmallVector<Operation *> producers;
  // find single-result and regionless op with i1 tensor result-type
  moduleOp.walk([&](Operation *op) {
    if (isValidI1Producer(op) && isPureAndRegionless(op)) {
      LOG_DEBUG("found i1 producer: " << *op << "\n");
      if (bm.getBlockIdByOp(op) != -1) {
        producers.push_back(op);
      }
    }
  });

  for (Operation *p : llvm::reverse(producers)) {
    SetVector<Operation *> consumers;
    for (OpOperand &use : p->getUses()) {
      Operation *consumer = use.getOwner();
      auto consumerInblock =
          CVPipeline::getAncestorInBlock(consumer, p->getBlock());
      if (!consumerInblock) {
        continue;
      }
      consumers.insert(consumerInblock);
    }
    if (consumers.empty()) {
      continue;
    }
    LOG_DEBUG("Producer: " << *p << "have " << consumers.size()
                           << " consumers.\n");
    LLVM_DEBUG(
        for (auto c : consumers) { LOG_DEBUG("consumer: " << *c << "\n"); });
    DenseSet<int> seenBlockIds;
    DenseMap<int, Operation *> blockId2Producer;
    auto orderedConsumuers = mlir::topologicalSort(consumers);
    // 1. Move into first consumer blockId
    int consumerBlockId = bm.getBlockIdByOp(orderedConsumuers[0]);
    if (consumerBlockId == -1) {
      // if i1 break by one control op, couldn't change.
      LOG_DEBUG("First consumer's blockid is -1.\n");
      seenBlockIds.insert(bm.getBlockIdByOp(p));
      blockId2Producer.insert({bm.getBlockIdByOp(p), p});
    } else if (bm.isSameBlock(p, orderedConsumuers[0])) {
      LOG_DEBUG("producer and consumer has same blockId.\n");
      seenBlockIds.insert(consumerBlockId);
      blockId2Producer.insert({consumerBlockId, p});
    } else {
      p->moveBefore(orderedConsumuers[0]);
      LOG_DEBUG("move producer " << *p << " to " << consumerBlockId << "\n");
      bm.updateBlockId(p, consumerBlockId);
      blockId2Producer.insert({consumerBlockId, p});
    }

    // 2. if there are other consumer, then clone producer.
    for (auto consumerInblock : orderedConsumuers) {
      int consumerBlockId = bm.getBlockIdByOp(consumerInblock);
      if (consumerBlockId == -1) {
        // if i1 break by one control op (scf.for/yield), couldn't change.
        continue;
      }
      if (!seenBlockIds.insert(consumerBlockId).second) {
        auto producer = blockId2Producer[consumerBlockId];
        for (auto info : llvm::enumerate(p->getResults())) {
          auto id = info.index();
          info.value().replaceUsesWithIf(
              producer->getResult(id), [&](OpOperand &use) {
                return use.getOwner() == consumerInblock;
              });
        }
        continue;
      }
      // Create one producer for now blockid.
      LOG_DEBUG("clone producer " << *p << " to " << consumerBlockId << "\n");
      auto cloned = OpBuilder(consumerInblock).clone(*p);
      blockId2Producer.insert({consumerBlockId, cloned});
      bm.updateBlockId(cloned, consumerBlockId);
      for (auto info : llvm::enumerate(p->getResults())) {
        auto id = info.index();
        info.value().replaceUsesWithIf(
            cloned->getResult(id),
            [&](OpOperand &use) { return use.getOwner() == consumerInblock; });
      }
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersIntoUsersPass() {
  return std::make_unique<SinkI1ProducersIntoUsersPass>();
}

} // namespace triton
} // namespace mlir
