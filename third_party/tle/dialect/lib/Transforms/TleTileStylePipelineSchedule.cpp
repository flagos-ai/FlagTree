// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLETILESTYLEPIPELINESCHEDULE
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static constexpr llvm::StringLiteral
    kTleTileStylePipelineAttr("tle.tile_style_pipeline");

static IntegerAttr getI32Attr(MLIRContext *ctx, int64_t value) {
  return IntegerAttr::get(IntegerType::get(ctx, 32), value);
}

static bool hasTwoStagePipelineAttr(scf::ForOp forOp) {
  auto attr = forOp->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName);
  return attr && attr.getInt() == 2;
}

static std::optional<int64_t> getScheduledMaxStage(scf::ForOp forOp) {
  auto attr = forOp->getAttrOfType<IntegerAttr>(tt::kScheduledMaxStageAttrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> getLoopStage(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool isLoadLikeProducer(Operation *op) {
  return isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op);
}

static ttg::LocalAllocOp getSingleDirectLocalAllocUser(tt::LoadOp loadOp) {
  ttg::LocalAllocOp localAlloc;
  for (Operation *user : loadOp->getUsers()) {
    auto allocOp = dyn_cast<ttg::LocalAllocOp>(user);
    if (!allocOp || allocOp.getSrc() != loadOp.getResult())
      return ttg::LocalAllocOp();
    if (localAlloc)
      return ttg::LocalAllocOp();
    localAlloc = allocOp;
  }
  return localAlloc;
}

static Value stripProducerMemDescViews(Value value) {
  Value current = value;
  while (true) {
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = subslice.getSrc();
      continue;
    }
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = index.getSrc();
      continue;
    }
    break;
  }
  return current;
}

static bool isDotLikeConsumer(Operation *op) {
  return isa<tt::DotOpInterface>(op);
}

static bool isTleOp(Operation *op) {
  return op && op->getName().getDialectNamespace() == "tle";
}

static bool isTileProducerViewLikeOp(Operation *op) {
  return isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(op) ||
         op->getName().getStringRef() == "tle.memdesc_wgmma_view";
}

static bool
hasTleMemdescWgmmaView(Value value,
                       llvm::SmallDenseSet<Operation *, 8> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (user->getName().getStringRef() == "tle.memdesc_wgmma_view")
      return true;
    if (isTileProducerViewLikeOp(user) && !user->getResults().empty() &&
        hasTleMemdescWgmmaView(user->getResult(0), visited))
      return true;
  }
  return false;
}

static bool hasDotLikeConsumer(Value value,
                               llvm::SmallDenseSet<Operation *, 8> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isDotLikeConsumer(user))
      return true;
    if (isTileProducerViewLikeOp(user) && !user->getResults().empty() &&
        hasDotLikeConsumer(user->getResult(0), visited))
      return true;
  }
  return false;
}

static bool isTleLegacyLoadAllocProducer(Operation *op) {
  auto loadOp = dyn_cast<tt::LoadOp>(op);
  if (!loadOp)
    return false;
  auto allocOp = getSingleDirectLocalAllocUser(loadOp);
  if (!allocOp)
    return false;
  llvm::SmallDenseSet<Operation *, 8> visited;
  return hasTleMemdescWgmmaView(allocOp.getResult(), visited);
}

static Operation *
getFirstDotLikeConsumerOp(Value value,
                          llvm::SmallDenseSet<Operation *, 8> &visited) {
  Operation *firstDot = nullptr;
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isDotLikeConsumer(user)) {
      if (!firstDot || user->isBeforeInBlock(firstDot))
        firstDot = user;
      continue;
    }
    if (!isTileProducerViewLikeOp(user) || user->getNumResults() == 0)
      continue;
    if (Operation *nested =
            getFirstDotLikeConsumerOp(user->getResult(0), visited)) {
      if (!firstDot || nested->isBeforeInBlock(firstDot))
        firstDot = nested;
    }
  }
  return firstDot;
}

static bool isAsyncCopyLikeProducer(Operation *op) {
  auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op);
  if (!asyncCopy)
    return false;
  if (!asyncCopy->hasAttr(kTleLocalPointerAsyncStoreAttr))
    return false;
  Value baseMemDesc = stripProducerMemDescViews(asyncCopy->getOperand(1));
  llvm::SmallDenseSet<Operation *, 8> visited;
  return baseMemDesc && hasDotLikeConsumer(baseMemDesc, visited);
}

static bool hasAnyLoopStageAttr(scf::ForOp forOp) {
  return llvm::any_of(
      forOp.getBody()->without_terminator(),
      [&](Operation &op) { return op.hasAttr(tt::kLoopStageAttrName); });
}

static Operation *findFirstDirectAsyncDotConsumer(scf::ForOp forOp) {
  Operation *firstDot = nullptr;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op);
    if (!asyncCopy)
      continue;
    Value baseMemDesc = stripProducerMemDescViews(asyncCopy->getOperand(1));
    if (!baseMemDesc)
      continue;
    llvm::SmallDenseSet<Operation *, 8> visited;
    Operation *dot = getFirstDotLikeConsumerOp(baseMemDesc, visited);
    if (!dot)
      continue;
    if (!firstDot || dot->isBeforeInBlock(firstDot))
      firstDot = dot;
  }
  return firstDot;
}

static bool isEligibleDirectAsyncLoop(scf::ForOp forOp) {
  if (!hasTwoStagePipelineAttr(forOp))
    return false;
  if (forOp->hasAttr(kTleTileStylePipelineAttr))
    return false;

  bool hasAsyncProducer = false;
  bool hasTleWgmmaViewConsumer = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op);
    if (!asyncCopy)
      continue;
    hasAsyncProducer |= isAsyncCopyLikeProducer(&op);
    Value baseMemDesc = stripProducerMemDescViews(asyncCopy->getOperand(1));
    if (!baseMemDesc)
      continue;
    llvm::SmallDenseSet<Operation *, 8> visited;
    hasTleWgmmaViewConsumer |= hasTleMemdescWgmmaView(baseMemDesc, visited);
  }
  if (!hasAsyncProducer || !hasTleWgmmaViewConsumer)
    return false;
  return findFirstDirectAsyncDotConsumer(forOp) != nullptr;
}

static bool isEligibleForTileStylePipeline(scf::ForOp forOp) {
  if (!hasTwoStagePipelineAttr(forOp))
    return false;
  if (forOp->hasAttr(kTleTileStylePipelineAttr))
    return false;
  auto scheduledMaxStage = getScheduledMaxStage(forOp);
  if (!scheduledMaxStage || *scheduledMaxStage != 1)
    return false;

  bool hasStage0Producer = false;
  bool hasStage1Consumer = false;
  bool hasStage0Dot = false;
  bool hasUnknownStage = false;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto stage = getLoopStage(&op);
    if (!stage) {
      hasUnknownStage = true;
      break;
    }
    if (*stage > 1)
      return false;
    if (*stage == 0) {
      hasStage0Producer |=
          isTleLegacyLoadAllocProducer(&op) || isAsyncCopyLikeProducer(&op);
      hasStage0Dot |= isDotLikeConsumer(&op);
    } else if (*stage == 1) {
      hasStage1Consumer |= isDotLikeConsumer(&op);
    }
  }

  if (hasUnknownStage || hasStage0Dot)
    return false;
  return hasStage0Producer && hasStage1Consumer;
}

static void synthesizeDirectAsyncStages(scf::ForOp forOp, Operation *firstDot) {
  auto ctx = forOp.getContext();
  for (Operation &op : forOp.getBody()->without_terminator()) {
    int64_t stage = op.isBeforeInBlock(firstDot) ? 0 : 2;
    op.setAttr(tt::kLoopStageAttrName, getI32Attr(ctx, stage));
    op.setAttr(tt::kLoopClusterAttrName, getI32Attr(ctx, 0));
  }
  forOp->setAttr(tt::kScheduledMaxStageAttrName, getI32Attr(ctx, 2));
  forOp->setAttr(kTleTileStylePipelineAttr, getI32Attr(ctx, 1));
}

class TileStylePipelineSchedulePass
    : public impl::TritonTleTileStylePipelineScheduleBase<
          TileStylePipelineSchedulePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp forOp) {
      if (isEligibleForTileStylePipeline(forOp))
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      for (Operation &op : forOp.getBody()->without_terminator()) {
        auto stage = getLoopStage(&op);
        if (!stage || *stage == 0)
          continue;
        op.setAttr(tt::kLoopStageAttrName,
                   getI32Attr(op.getContext(), *stage + 1));
      }
      forOp->setAttr(tt::kScheduledMaxStageAttrName,
                     getI32Attr(forOp.getContext(), 2));
      forOp->setAttr(kTleTileStylePipelineAttr,
                     getI32Attr(forOp.getContext(), 1));
    }

    module.walk([&](scf::ForOp forOp) {
      if (!isEligibleDirectAsyncLoop(forOp))
        return;
      Operation *firstDot = findFirstDirectAsyncDotConsumer(forOp);
      if (!firstDot)
        return;
      synthesizeDirectAsyncStages(forOp, firstDot);
    });
  }
};

} // namespace
} // namespace mlir::triton::tle
