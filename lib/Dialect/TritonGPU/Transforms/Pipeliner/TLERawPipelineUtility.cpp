// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT

#ifdef __TLE__

#include "TLERawPipelineUtility.h"

#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tle = mlir::triton::tle;

namespace mlir::triton::gpu {
namespace {

static constexpr llvm::StringLiteral TLERawPipelineHint("pipeline");

struct OutputInfo {
  unsigned resultIndex;
  unsigned operandIndex;
  Value originalAlloc;
  ttg::MemDescType type;
  Value ring;
};

struct RawInfo {
  tle::DSLRegionOp op;
  int depth = 0;
  SmallVector<OutputInfo> outputs;
  Value insertIdx;
  Value extractIdx;
};

static bool isOutputDealloc(Operation *user, Value alloc) {
  auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user);
  return dealloc && dealloc.getSrc() == alloc;
}

static bool reachesWGMMAThroughViews(Value value) {
  SmallVector<Value> worklist{value};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (isa<ttng::WarpGroupDotOp>(user))
        return true;
      // A local_load ends the lifetime of the raw shared slot.  Any later
      // local_alloc used by WGMMA is a different allocation and is protected
      // by the native WGMMA wait scheduling.
      if (user->hasTrait<OpTrait::MemDescViewTrait>())
        worklist.append(user->getResults().begin(), user->getResults().end());
    }
  }
  return false;
}

static FailureOr<RawInfo> analyze(tle::DSLRegionOp op, scf::ForOp forOp,
                                  CoarseSchedule &schedule) {
  RawInfo info;
  info.op = op;
  if (op->getParentOp() != forOp)
    return op.emitError("hint=\"pipeline\" requires tle.dsl_region to be "
                        "directly inside scf.for");
  if (!schedule.count(op))
    return op.emitError("hint=\"pipeline\" did not receive a native "
                        "software-pipeline stage");

  ArrayRef<int32_t> indices = op.getOutputOperandIndices();
  if (indices.empty() || indices.size() != op.getNumResults())
    return op.emitError("hint=\"pipeline\" requires one output operand "
                        "index for every tle.dsl_region result");

  int producerStage = schedule[op].first;
  int lastUseStage = producerStage;
  bool needsWGMMAExtraSlot = false;
  DenseSet<unsigned> outputOperands;
  for (auto [resultIndex, signedOperandIndex] : llvm::enumerate(indices)) {
    if (signedOperandIndex < 0 ||
        static_cast<unsigned>(signedOperandIndex) >= op.getNumOperands())
      return op.emitError("contains an invalid output operand index");
    unsigned operandIndex = static_cast<unsigned>(signedOperandIndex);
    if (!outputOperands.insert(operandIndex).second)
      return op.emitError(
          "pipeline results must alias distinct output operands");
    Value alloc = op.getOperand(operandIndex);
    auto operandTy = dyn_cast<ttg::MemDescType>(alloc.getType());
    auto resultTy =
        dyn_cast<ttg::MemDescType>(op.getResult(resultIndex).getType());
    if (!operandTy || !resultTy || operandTy != resultTy)
      return op.emitError("pipeline outputs must be identically typed "
                          "shared-memory memdescs");
    if (!alloc.getDefiningOp<ttg::LocalAllocOp>() ||
        alloc.getDefiningOp()->getParentOp() != forOp)
      return op.emitError(
          "pipeline outputs must use loop-local ttg.local_alloc");

    for (Operation *user : alloc.getUsers()) {
      if (user == op || isa<ttg::LocalStoreOp>(user) ||
          isOutputDealloc(user, alloc))
        continue;
      return user->emitError("unsupported direct use of a TLE-Raw pipeline "
                             "output allocation");
    }

    for (Operation *user : op.getResult(resultIndex).getUsers()) {
      Operation *top = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!top || !schedule.count(top))
        return user->emitError("TLE-Raw pipeline result escapes its loop");
      lastUseStage = std::max(lastUseStage, schedule[top].first);
    }
    needsWGMMAExtraSlot |= reachesWGMMAThroughViews(op.getResult(resultIndex));
    info.outputs.push_back({static_cast<unsigned>(resultIndex),
                            operandIndex,
                            alloc,
                            operandTy,
                            {}});
  }

  info.depth = lastUseStage - producerStage;
  if (info.depth <= 0)
    return op.emitError("hint=\"pipeline\" requires a consumer in a later "
                        "native pipeline stage");
  if (needsWGMMAExtraSlot)
    ++info.depth;
  return info;
}

static void eraseScheduledOp(CoarseSchedule &schedule, Operation *op) {
  if (schedule.count(op))
    schedule.erase(op);
  op->erase();
}

} // namespace

bool isTLERawPipelineOp(Operation *op) {
  auto region = dyn_cast_or_null<tle::DSLRegionOp>(op);
  if (!region)
    return false;
  auto hint = region->getAttrOfType<StringAttr>("hint");
  return hint && hint.getValue() == TLERawPipelineHint;
}

LogicalResult validateTLERawPipelineOps(ModuleOp moduleOp) {
  WalkResult result = moduleOp.walk([&](Operation *candidate) {
    if (!isTLERawPipelineOp(candidate))
      return WalkResult::advance();

    auto op = cast<tle::DSLRegionOp>(candidate);
    auto forOp = dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!forOp) {
      op.emitError("hint=\"pipeline\" requires tle.dsl_region to be "
                   "directly inside scf.for");
      return WalkResult::interrupt();
    }

    CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp))) {
      forOp.emitError("contains hint=\"pipeline\" tle.dsl_region but native "
                      "software pipelining produced no schedule");
      return WalkResult::interrupt();
    }
    if (failed(analyze(op, forOp, schedule)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

FailureOr<scf::ForOp> lowerTLERawPipelineOps(scf::ForOp forOp,
                                             CoarseSchedule &schedule) {
  SmallVector<RawInfo> raws;
  for (Operation &candidate : forOp.getBody()->without_terminator()) {
    if (!isTLERawPipelineOp(&candidate))
      continue;
    auto info = analyze(cast<tle::DSLRegionOp>(&candidate), forOp, schedule);
    if (failed(info))
      return failure();
    raws.push_back(std::move(*info));
  }
  if (raws.empty())
    return forOp;

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);

  for (RawInfo &raw : raws) {
    for (OutputInfo &output : raw.outputs) {
      auto ringTy = triton::getMultiBufferedType(output.type, raw.depth);
      output.ring = ttg::LocalAllocOp::create(builder, raw.op.getLoc(), ringTy);
      OpBuilder after(forOp);
      after.setInsertionPointAfter(forOp);
      ttg::LocalDeallocOp::create(after, raw.op.getLoc(), output.ring);
    }
  }

  unsigned firstNewArg = forOp.getBody()->getNumArguments();
  SmallVector<Value> counterInits;
  for (size_t i = 0; i < raws.size(); ++i) {
    counterInits.push_back(minusOne);
    counterInits.push_back(minusOne);
  }
  forOp = addIterArgsToLoop(builder, forOp, counterInits);
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  yield.getResultsMutable().append(counterInits);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  unsigned arg = firstNewArg;
  for (RawInfo &raw : raws) {
    Value depth = arith::ConstantIntOp::create(builder, loc, raw.depth, 32);
    raw.insertIdx = createIncrementModulo(
        builder, loc, forOp.getBody()->getArgument(arg++), depth, zero, one);
    raw.extractIdx = createIncrementModulo(
        builder, loc, forOp.getBody()->getArgument(arg++), depth, zero, one);
  }

  for (RawInfo &raw : raws) {
    for (OutputInfo &output : raw.outputs) {
      Operation *insertAnchor = raw.op;
      for (Operation *user : output.originalAlloc.getUsers()) {
        if (isa<ttg::LocalStoreOp>(user) &&
            user->getBlock() == raw.op->getBlock() &&
            user->isBeforeInBlock(insertAnchor))
          insertAnchor = user;
      }
      OpBuilderForStage producerBuilder(raw.op.getLoc(), forOp, schedule);
      producerBuilder.setInsertionPoint(insertAnchor);
      producerBuilder.setStageCluster(schedule[insertAnchor]);
      Value insertView =
          createSingleBufferView(producerBuilder, output.ring, raw.insertIdx);

      SmallVector<Operation *> oldAllocUsers(output.originalAlloc.getUsers());
      for (Operation *user : oldAllocUsers) {
        if (user == raw.op)
          continue;
        if (auto store = dyn_cast<ttg::LocalStoreOp>(user)) {
          store.getDstMutable().assign(insertView);
          continue;
        }
        if (isOutputDealloc(user, output.originalAlloc))
          eraseScheduledOp(schedule, user);
      }
      raw.op->setOperand(output.operandIndex, insertView);

      Operation *firstUse =
          getFirstUseOfPipelinedOp({raw.op.getOperation()}, forOp, schedule);
      if (!firstUse)
        return raw.op.emitError("pipeline output has no in-loop consumer");
      OpBuilderForStage consumerBuilder(raw.op.getLoc(), forOp, schedule);
      consumerBuilder.setInsertionPoint(firstUse);
      consumerBuilder.setStageCluster(schedule[firstUse]);
      Value extractView =
          createSingleBufferView(consumerBuilder, output.ring, raw.extractIdx);
      raw.op.getResult(output.resultIndex).replaceAllUsesWith(extractView);

      Operation *oldAlloc = output.originalAlloc.getDefiningOp();
      if (!output.originalAlloc.use_empty())
        return oldAlloc->emitError(
            "failed to redirect all pipeline buffer uses");
      eraseScheduledOp(schedule, oldAlloc);
    }
  }

  unsigned yieldIndex = firstNewArg - 1;
  for (RawInfo &raw : raws) {
    yield.setOperand(yieldIndex++, raw.insertIdx);
    yield.setOperand(yieldIndex++, raw.extractIdx);
  }

  scheduleDependencies(forOp, schedule);
  return forOp;
}

Operation *predicateTLERawPipelineOp(RewriterBase &rewriter, Operation *op,
                                     Value pred) {
  auto region = dyn_cast<tle::DSLRegionOp>(op);
  if (!region || !isTLERawPipelineOp(op))
    return nullptr;
  rewriter.setInsertionPoint(region);
  auto ifOp = scf::IfOp::create(rewriter, region.getLoc(),
                                region.getResultTypes(), pred,
                                /*withElseRegion=*/true);

  OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
  auto thenYield =
      scf::YieldOp::create(thenBuilder, region.getLoc(), region.getResults());
  region->moveBefore(thenYield);

  SmallVector<Value> inactiveResults;
  for (int32_t operandIndex : region.getOutputOperandIndices())
    inactiveResults.push_back(region.getOperand(operandIndex));
  OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
  scf::YieldOp::create(elseBuilder, region.getLoc(), inactiveResults);

  for (auto [oldResult, newResult] :
       llvm::zip(region.getResults(), ifOp.getResults())) {
    oldResult.replaceUsesWithIf(
        newResult, [&](OpOperand &use) { return use.getOwner() != thenYield; });
  }
  return ifOp;
}

} // namespace mlir::triton::gpu

#endif // __TLE__
