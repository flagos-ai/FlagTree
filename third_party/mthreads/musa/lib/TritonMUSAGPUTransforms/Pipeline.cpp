#include "Dialect/MUSA/IR/Dialect.h"
#include "SqmmaPipelineUtils.h"
#include "TMEPipelineUtils.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType))
    mask = tt::SplatOp::create(rewriter, loc, maskType, pred);
  if (currentMask)
    mask = arith::AndIOp::create(rewriter, loc, mask, currentMask);
  return mask;
}

static Operation *predicateVoidOpWithIf(RewriterBase &rewriter, Operation *op,
                                        Value pred) {
  if (isConstantIntValue(pred, 1))
    return op;
  if (!op->getResults().empty()) {
    op->emitOpError("MUSA pipeliner can only branch-predicate void ops");
    llvm::report_fatal_error("Fatal pipeliner error");
  }
  rewriter.setInsertionPoint(op);
  auto ifOp = scf::IfOp::create(rewriter, op->getLoc(), pred, false);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  Operation *cloned = rewriter.clone(*op);
  rewriter.eraseOp(op);
  return cloned;
}

static Operation *musaPredicateOp(RewriterBase &rewriter, Operation *op,
                                  Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isConstantIntValue(pred, 1))
    return op;
  if (isa<LLVM::AssumeOp>(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return op;
  if (op->hasTrait<OpTrait::LocalLoadTrait>())
    return op;
  if (isa<ttg::LocalStoreOp>(op))
    return op;
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    rewriter.setInsertionPoint(op);
    Value cnd = getPredMask(rewriter, ifOp.getCondition().getType(),
                            ifOp.getCondition(), pred);
    ifOp.getConditionMutable().assign(cnd);
    return op;
  }
  if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(asyncCopyOp);
    Value mask = getPredMask(rewriter, asyncCopyOp.getSrc().getType(),
                             asyncCopyOp.getMask(), pred);
    asyncCopyOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto copyOp = dyn_cast<triton::musa::AsyncTMECopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(copyOp);
    Value mask = getPredMask(rewriter, copyOp.getPred().getType(),
                             copyOp.getPred(), pred);
    copyOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto copyOp = dyn_cast<triton::musa::AsyncTMECopyLocalToGlobalOp>(op)) {
    rewriter.setInsertionPoint(copyOp);
    Value mask = getPredMask(rewriter, copyOp.getPred().getType(),
                             copyOp.getPred(), pred);
    copyOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto addTransOp = dyn_cast<triton::musa::BarrierAddTransOp>(op)) {
    rewriter.setInsertionPoint(addTransOp);
    Value mask = getPredMask(rewriter, addTransOp.getPred().getType(),
                             addTransOp.getPred(), pred);
    addTransOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto arriveOp = dyn_cast<triton::musa::ArriveBarrierNoRetOp>(op)) {
    rewriter.setInsertionPoint(arriveOp);
    Value mask = getPredMask(rewriter, arriveOp.getPred().getType(),
                             arriveOp.getPred(), pred);
    arriveOp.getPredMutable().assign(mask);
    return op;
  }
  if (isa<triton::musa::WaitBarrierOp, triton::musa::InitArrivalOp,
          triton::musa::TMEStoreCommitOp, triton::musa::TMEStoreReadWaitOp>(
          op)) {
    return predicateVoidOpWithIf(rewriter, op, pred);
  }
  if (!op->isRegistered())
    return op;

  op->emitOpError("MUSA pipeliner doesn't know how to predicate this op.");
  llvm::report_fatal_error("Fatal pipeliner error");
}

static Operation *musaWrapInMaskOp(RewriterBase &rewriter, Operation *op,
                                   Value pred) {
  auto mask =
      ttg::MaskOp::create(rewriter, op->getLoc(), op->getResultTypes(), pred);
  rewriter.createBlock(&mask->getRegion(0));
  rewriter.setInsertionPointToStart(&mask->getRegion(0).front());
  auto newOp = rewriter.clone(*op);
  ttg::MaskReturnOp::create(rewriter, op->getLoc(), newOp->getResults());
  op->replaceAllUsesWith(mask->getResults());
  rewriter.eraseOp(op);
  return mask;
}

static void musaResolveMaskOp(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp);

  auto arithDialect =
      moduleOp.getContext()->getLoadedDialect<arith::ArithDialect>();
  RewritePatternSet patterns(moduleOp.getContext());
  arithDialect->getCanonicalizationPatterns(patterns);
  if (applyPatternsGreedily(moduleOp, std::move(patterns)).failed())
    llvm::report_fatal_error("Failed to canonicalize the IR");

  SmallVector<ttg::MaskOp> maskOps;
  moduleOp.walk([&](ttg::MaskOp maskOp) { maskOps.push_back(maskOp); });
  for (ttg::MaskOp maskOp : maskOps) {
    rewriter.setInsertionPoint(maskOp);
    while (&maskOp.getBody()->front() != maskOp.getBody()->getTerminator()) {
      Operation *op = &maskOp.getBody()->front();
      rewriter.moveOpBefore(op, maskOp);
      (void)musaPredicateOp(rewriter, op, maskOp.getPred());
    }
    maskOp->replaceAllUsesWith(
        maskOp.getBody()->getTerminator()->getOperands());
    maskOp->erase();
  }
}

// =========================
// Lower Loads / Descriptors
// =========================

static bool mustLoadToRegisters(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
      return true;
  }

  if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(op)) {
    return failed(
        triton::musa::resolveDescriptorLoadLandingMemDescType(descLoad));
  }

  if (!op->hasOneUse())
    return true;
  auto alloc = dyn_cast<ttg::LocalAllocOp>(*op->getUsers().begin());
  if (!alloc)
    return true;

  Attribute loadEncoding;
  if (auto descGather = dyn_cast<tt::DescriptorGatherOp>(op)) {
    loadEncoding = ttng::getEncodingFromDescriptor(op, descGather.getType(),
                                                   descGather.getDesc());
  }
  return loadEncoding && (loadEncoding != alloc.getType().getEncoding());
}

static ttg::MemDescType
getGenericMusaLoadMemDescType(Operation *loadOp, ttg::SharedEncodingTrait enc) {
  auto tensorTy = cast<RankedTensorType>(loadOp->getResultTypes().front());
  auto sharedSpace = ttg::SharedMemorySpaceAttr::get(loadOp->getContext());
  return ttg::MemDescType::get(tensorTy.getShape(), tensorTy.getElementType(),
                               enc, sharedSpace, /*mutableMemory=*/true);
}

static ttg::MemDescType getMusaMultiBufferedType(ttg::MemDescType viewTy,
                                                 int32_t depth) {
  SmallVector<int64_t> shape(viewTy.getShape().begin(),
                             viewTy.getShape().end());
  SmallVector<int64_t> allocShape(viewTy.getAllocShape().begin(),
                                  viewTy.getAllocShape().end());
  shape.insert(shape.begin(), depth);
  allocShape.insert(allocShape.begin(), depth);
  return ttg::MemDescType::get(shape, viewTy.getElementType(),
                               viewTy.getEncoding(), viewTy.getMemorySpace(),
                               /*mutableMemory=*/true, allocShape);
}

static Value createMusaDescriptorAlloc(scf::ForOp forOp, Location loc,
                                       ttg::MemDescType landingTy,
                                       unsigned distance) {
  OpBuilder builder(forOp);
  auto allocTy = getMusaMultiBufferedType(landingTy, distance);
  Value alloc = ttg::LocalAllocOp::create(builder, loc, allocTy);
  builder.setInsertionPointAfter(forOp);
  ttg::LocalDeallocOp::create(builder, loc, alloc);
  return alloc;
}

static TypedValue<ttg::MemDescType>
createMusaSingleBufferView(OpBuilder &builder, Value alloc, Value idx) {
  auto allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<int64_t> viewShape(allocTy.getShape().drop_front().begin(),
                                 allocTy.getShape().drop_front().end());
  SmallVector<int64_t> viewAllocShape(
      allocTy.getAllocShape().drop_front().begin(),
      allocTy.getAllocShape().drop_front().end());
  auto viewTy = ttg::MemDescType::get(
      viewShape, allocTy.getElementType(), allocTy.getEncoding(),
      allocTy.getMemorySpace(), allocTy.getMutableMemory(), viewAllocShape);
  return ttg::MemDescIndexOp::create(builder, alloc.getLoc(), viewTy, alloc,
                                     idx);
}

static bool hasMusaSqmmaRoot(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->without_terminator(), [](Operation &op) {
    return isa<triton::musa::SquadDotOp>(op);
  });
}

static bool hasMusaDescriptorLoadRoot(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->without_terminator(), [](Operation &op) {
    return isa<tt::DescriptorLoadOp>(op);
  });
}

static llvm::MapVector<Operation *, std::pair<int, Operation *>>
loadOpsToMusaSqmmaIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                                   int numStages) {
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel;
  DenseSet<Operation *> seen;
  DenseSet<Operation *> excluded;

  std::function<void(Operation *, Operation *, int)> dfs =
      [&](Operation *op, Operation *finalUser, int distance) {
        if (!seen.insert(op).second || excluded.count(op))
          return;
        if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
          if (loadOpToIndLevel.count(op)) {
            int level = loadOpToIndLevel[op].first;
            if (level != distance) {
              loadOpToIndLevel.erase(op);
              excluded.insert(op);
              return;
            }
          } else {
            loadOpToIndLevel[op] = {distance, finalUser};
          }
          finalUser = op;
          distance++;
        }

        for (Value operand : getNestedOperands(op)) {
          if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
            if (operand == dotOp->getOperand(2))
              continue;
          }
          Operation *defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock())
            dfs(defOp, finalUser, distance);
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isa<triton::musa::SquadDotOp>(op))
      continue;
    seen.clear();
    dfs(&op, &op, 0);
  }

  if (pipelineWithoutDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
        dfs(&op, &op, 0);
    }
  }

  for (auto iter = loadOpToIndLevel.begin(); iter != loadOpToIndLevel.end();) {
    if (iter->second.first >= numStages - 1)
      iter = loadOpToIndLevel.erase(iter);
    else
      ++iter;
  }

  return loadOpToIndLevel;
}

static tt::CoarseSchedule::Cluster scheduleMusaSqmmaPrologueAndEpilogue(
    scf::ForOp forOp, tt::CoarseSchedule &schedule,
    DenseSet<Operation *> &rootUsers, int numStages) {
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  DenseMap<scf::IfOp, int> ifsToStage;
  for (int stage = 0; stage < numStages; ++stage) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      llvm::SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.omitUsesFromAbove = false;
      (void)getBackwardSlice(op, &backwardSlice, opt);
      for (Operation *sliceOp : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(sliceOp))
          ifsToStage.insert({ifOp, stage});
      }
    }
  }
  if (!ifsToStage.empty()) {
    tt::CoarseSchedule::Cluster prologueCluster =
        schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage)
      schedule.insert(ifOp, stage, prologueCluster);
  }

  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (Operation *rootUser : rootUsers) {
    llvm::SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);
    int stage = schedule[rootUser].first;
    for (Operation *sliceOp : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(sliceOp);
      if (!ifOp) {
        Operation *parentOp = sliceOp->getParentOp();
        if (parentOp && parentOp->getParentOp() == forOp.getOperation())
          ifOp = dyn_cast<scf::IfOp>(parentOp);
      }
      if (ifOp)
        schedule.insertIfAbsent(ifOp, stage, epilogueCluster);
    }
  }

  return afterPrologue;
}

static void scheduleMusaSqmmaDependencies(scf::ForOp forOp,
                                          tt::CoarseSchedule &schedule,
                                          int numStages) {
  auto opsInOrder = schedule.getOpsInOrder(forOp);
  for (int stage = 0; stage < numStages; ++stage) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, /*includeArg=*/false);
    }
  }
}

static void scheduleMusaSqmmaDistanceOneDependencies(
    scf::ForOp forOp, tt::CoarseSchedule &schedule, int numStages) {
  DenseMap<tt::CoarseSchedule::ClusterHash, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getArgNumber() == 0 || arg.getOwner() != op.getBlock())
        continue;
      auto yieldOp = op.getBlock()->getTerminator();
      Value yielded = yieldOp->getOperand(arg.getArgNumber() - 1);
      Operation *defOp = yielded.getDefiningOp();
      if (!defOp || schedule.count(defOp))
        continue;
      if (isa<tt::LoadOp>(defOp)) {
        schedule.insertIfAbsent(defOp, stage, cluster);
        schedule.insertDepsOfOp(defOp, stage, cluster, /*includeArg=*/true,
                                /*insertIfEarlier=*/true);
        continue;
      }
      auto clusterHash = tt::CoarseSchedule::hashCluster(cluster);
      if (!dist1Cluster.count(clusterHash))
        dist1Cluster[clusterHash] = schedule.clusters.newBefore(cluster);
      schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[clusterHash]);
      schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[clusterHash],
                              /*includeArg=*/true,
                              /*insertIfEarlier=*/true);
    }
  }
}

static void scheduleMusaSqmmaRemainingToLastStage(
    scf::ForOp forOp, tt::CoarseSchedule &schedule,
    tt::CoarseSchedule::Cluster afterPrologue, int numStages) {
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      opToCluster[&op] = afterPrologue;
  }

  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    if (stage == numStages - 1)
      queue.push_back(op);
  }

  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (Operation *user : op->getUsers()) {
      if (!opToCluster.count(user))
        continue;
      auto userCluster = opToCluster[user];
      auto opCluster =
          schedule.count(op) ? schedule[op].second : opToCluster[op];
      if (*userCluster < *opCluster) {
        opToCluster[user] = opCluster;
        queue.push_back(user);
      }
    }
  }

  for (auto [op, cluster] : opToCluster)
    schedule.insert(op, numStages - 1, cluster);
}

static FailureOr<tt::CoarseSchedule>
synthesizeMusaSqmmaSchedule(scf::ForOp forOp, int defaultNumStages) {
  if (!triton::gpu::isSafeToPipeline(forOp))
    return failure();
  if (!hasMusaSqmmaRoot(forOp))
    return failure();

  int numStages = tt::getNumStagesOrDefault(forOp, defaultNumStages);
  if (numStages <= 1)
    return failure();

  auto loadOpToIndLevel = loadOpsToMusaSqmmaIndirectionLevel(
      forOp, forOp->hasAttr(tt::kNumStagesAttrName), numStages);
  if (loadOpToIndLevel.empty())
    return failure();

  tt::CoarseSchedule schedule(numStages);
  DenseSet<Operation *> rootUsers;
  int maxIndirectionLevel = -1;
  for (auto &[loadOp, info] : loadOpToIndLevel)
    maxIndirectionLevel = std::max(maxIndirectionLevel, info.first);
  if (maxIndirectionLevel < 0)
    return failure();

  auto rootUsersCluster = schedule.clusters.newAtFront();
  for (auto &[loadOp, info] : loadOpToIndLevel) {
    Operation *use = info.second;
    if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }
  if (rootUsers.empty())
    return failure();

  unsigned stagesBetweenLoads = 0;
  if (numStages > 2) {
    stagesBetweenLoads =
        (static_cast<unsigned>(numStages - 2 + maxIndirectionLevel)) /
        static_cast<unsigned>(maxIndirectionLevel + 1);
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; ++i)
    loadClusters.push_back(schedule.clusters.newAtBack());

  for (auto &[loadOp, info] : loadOpToIndLevel) {
    int stage = (maxIndirectionLevel - info.first) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadClusters[info.first]);
  }

  auto afterPrologue = scheduleMusaSqmmaPrologueAndEpilogue(
      forOp, schedule, rootUsers, numStages);
  scheduleMusaSqmmaDependencies(forOp, schedule, numStages);
  scheduleMusaSqmmaDistanceOneDependencies(forOp, schedule, numStages);
  scheduleMusaSqmmaRemainingToLastStage(forOp, schedule, afterPrologue,
                                        numStages);
  return schedule;
}

static int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                              tt::CoarseSchedule &schedule) {
  assert(schedule.count(op) && "Op not found in the schedule");
  int defStage = schedule[op].first;
  tt::CoarseSchedule::Cluster defCluster = schedule[op].second;
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers =
      triton::getTopLevelUsersInLoop(op, forOp);
  if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
    DenseSet<Operation *> allocUsers;
    for (Operation *topLevelUser : topLevelUsers) {
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(topLevelUser)) {
        DenseSet<Operation *> users =
            triton::getTopLevelUsersInLoop(localAlloc, forOp);
        allocUsers.insert(users.begin(), users.end());
      }
    }
    topLevelUsers.insert(allocUsers.begin(), allocUsers.end());
  }
  DenseSet<Operation *> topLevelWaitUsers;
  for (Operation *topLevelUser : topLevelUsers) {
    if (isa<triton::musa::WaitBarrierOp>(topLevelUser))
      topLevelWaitUsers.insert(topLevelUser);
  }
  for (Operation *topLevelUser : topLevelUsers) {
    if (!schedule.count(topLevelUser)) {
      topLevelUser->emitOpError("top-level user missing from MUSA pipeline "
                                "schedule");
      llvm::report_fatal_error("Fatal pipeliner error");
    }
    int curUseStage = schedule[topLevelUser].first;
    tt::CoarseSchedule::Cluster curUseCluster = schedule[topLevelUser].second;
    if (*curUseCluster > *defCluster)
      curUseStage++;
    useStage = std::min(curUseStage, useStage.value_or(curUseStage));
  }
  for (Operation *topLevelUser : topLevelWaitUsers) {
    int curUseStage = schedule[topLevelUser].first;
    useStage = std::max(curUseStage, useStage.value_or(curUseStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "Op used before defined");
  return useStage.value() - defStage;
}

static Value createAlloc(scf::ForOp forOp, Operation *loadOp,
                         ttg::MemDescType landingTy, unsigned distance) {
  return createMusaDescriptorAlloc(forOp, loadOp->getLoc(), landingTy,
                                   distance);
}

static ttg::LocalAllocOp getSqmmaOperandLocalAllocUser(tt::LoadOp loadOp) {
  if (mustLoadToRegisters(loadOp))
    return {};
  auto localAlloc = dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
  if (!localAlloc ||
      !triton::musa::hasSqmmaOpIdxAttr(localAlloc.getOperation())) {
    return {};
  }
  return localAlloc;
}

static ttg::MemDescType getSqmmaOperandLandingMemDescType(tt::LoadOp loadOp) {
  auto localAlloc = getSqmmaOperandLocalAllocUser(loadOp);
  if (!localAlloc)
    return {};
  return dyn_cast<ttg::MemDescType>(localAlloc.getType());
}

static void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                            Value insertIdx, Value extractIdx, int contiguity,
                            tt::CoarseSchedule &schedule) {
  tt::OpBuilderForStage builder(loadOp.getLoc(), forOp, schedule);
  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  Value view = createMusaSingleBufferView(builder, alloc, insertIdx);
  auto sqmmaLocalAlloc = getSqmmaOperandLocalAllocUser(loadOp);
  if (sqmmaLocalAlloc)
    triton::musa::copySqmmaAttrs(sqmmaLocalAlloc.getOperation(),
                                 view.getDefiningOp());
  Operation *copy = ttg::AsyncCopyGlobalToLocalOp::create(
      builder, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile(), contiguity);
  Operation *commit =
      ttg::AsyncCommitGroupOp::create(builder, copy->getResult(0));

  builder.setStageCluster(schedule[firstUse]);
  auto wait = ttg::AsyncWaitOp::create(builder, commit->getResult(0), 0);
  auto viewLoad = createMusaSingleBufferView(builder, alloc, extractIdx);
  if (sqmmaLocalAlloc)
    triton::musa::copySqmmaAttrs(sqmmaLocalAlloc.getOperation(),
                                 viewLoad.getDefiningOp());

  if (sqmmaLocalAlloc) {
    replaceUsesAndPropagateType(builder, sqmmaLocalAlloc, viewLoad);
    sqmmaLocalAlloc.erase();
  } else if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {
    replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad,
                             wait.getResult());
  } else if (loadOp->use_begin() != loadOp->use_end()) {
    auto sharedLoad = ttg::LocalLoadOp::create(builder, loadOp.getType(),
                                               viewLoad, wait.getResult());
    auto select =
        arith::SelectOp::create(builder, loadOp.getType(), loadOp.getMask(),
                                sharedLoad.getResult(), other);
    loadOp->replaceAllUsesWith(select->getResults());
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

static void replaceDescriptorUsesWithMemDescOrLocalLoad(
    tt::DescriptorLoadOp loadOp, Value sourceMemDesc, RewriterBase &rewriter) {
  SmallVector<Operation *> users(loadOp->getUsers().begin(),
                                 loadOp->getUsers().end());
  Value tensorValue = loadOp.getResult();
  Value localLoadValue;
  auto getLocalLoadValue = [&]() -> Value {
    if (!localLoadValue) {
      localLoadValue = ttg::LocalLoadOp::create(rewriter, loadOp.getLoc(),
                                                loadOp.getType(), sourceMemDesc)
                           .getResult();
    }
    return localLoadValue;
  };

  for (Operation *user : users) {
    bool isTensorViewUser = isa<tt::TransOp, tt::ReshapeOp>(user);
    if (triton::musa::tryReplaceTensorUserWithMemDesc(rewriter, tensorValue,
                                                      sourceMemDesc, user) &&
        !isTensorViewUser)
      continue;
    if (isTensorViewUser && user->use_empty()) {
      rewriter.eraseOp(user);
      continue;
    }
    rewriter.setInsertionPoint(user);
    user->replaceUsesOfWith(tensorValue, getLocalLoadValue());
    if (isTensorViewUser && user->use_empty())
      rewriter.eraseOp(user);
  }
}

struct AsyncLoad {
  int stageDiff;
  int contiguity = 1;
  Value alloc;
  Value barrier;
  Operation *waitOp = nullptr;
  ttg::MemDescType landingMemDescTy;
};

struct LoadGroupInfo {
  Value insertIdx;
  Value extractIdx;
  Value phase;
  Value yieldPhase;
  bool hasTMALoad = false;
  int32_t barrierBase = 0;
};

static void scheduleScalarPipelineValue(tt::CoarseSchedule &schedule, Value v,
                                        int stage,
                                        tt::CoarseSchedule::Cluster cluster) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return;
  schedule.insert(op, stage, cluster);
  schedule.insertDepsOfOp(op, stage, cluster, /*includeArg=*/false,
                          /*insertIfEarlier=*/true);
}

static void collectNonViewUsers(Operation *op,
                                SmallVectorImpl<Operation *> &out) {
  for (Operation *user : op->getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      collectNonViewUsers(user, out);
      continue;
    }
    out.push_back(user);
  }
}

static LogicalResult
validateMusaScheduleClusters(scf::ForOp forOp, tt::CoarseSchedule &schedule) {
  DenseSet<tt::CoarseSchedule::ClusterHash> validClusters;
  for (auto it = schedule.clusters.begin(); it != schedule.clusters.end(); ++it)
    validClusters.insert(tt::CoarseSchedule::hashCluster(it));

  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto it = schedule.find(&op);
    if (it == schedule.end())
      continue;
    auto [stage, cluster] = it->second;
    auto clusterHash = tt::CoarseSchedule::hashCluster(cluster);
    if (validClusters.count(clusterHash))
      continue;
    op.emitError() << "scheduled into foreign cluster hash=" << clusterHash
                   << " at stage " << stage;
    return failure();
  }
  return success();
}

static Value materializeBarrierIdValue(OpBuilder &builder, Location loc,
                                       int32_t base, Value slotIdx) {
  Value baseVal = arith::ConstantIntOp::create(builder, loc, base, 32);
  if (auto constIdx = slotIdx.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(constIdx.getValueAttr());
    if (intAttr)
      return arith::ConstantIntOp::create(builder, loc, base + intAttr.getInt(),
                                          32);
  }
  return arith::AddIOp::create(builder, loc, baseVal, slotIdx);
}

static void convertScalarToTensorLoad(Operation *op,
                                      tt::CoarseSchedule &schedule,
                                      scf::ForOp forOp) {
  auto scalarLoad = cast<tt::LoadOp>(op);
  Type scalarTy = scalarLoad.getType();
  tt::OpBuilderForStage builder(op->getLoc(), op, schedule);
  builder.setInsertionPoint(op);
  MLIRContext *ctx = op->getContext();
  auto nWarps = ttg::lookupNumWarps(op->getParentRegion());
  ModuleOp mod = forOp->getParentOfType<ModuleOp>();
  auto threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  auto numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  auto blockedEnc =
      ttg::getDefaultBlockedEncoding(ctx, {1}, nWarps, threadsPerWarp, numCTAs);
  auto newPtrTy =
      RankedTensorType::get({1}, scalarLoad.getPtr().getType(), blockedEnc);
  auto newPtr =
      tt::SplatOp::create(builder, op->getLoc(), newPtrTy, scalarLoad.getPtr());
  scalarLoad.getPtrMutable().assign(newPtr);
  if (scalarLoad.getMask()) {
    auto newMaskTy =
        RankedTensorType::get({1}, scalarLoad.getMask().getType(), blockedEnc);
    auto newMask = tt::SplatOp::create(builder, op->getLoc(), newMaskTy,
                                       scalarLoad.getMask());
    scalarLoad.getMaskMutable().assign(newMask);
  }
  if (scalarLoad.getOther()) {
    auto newOtherTy =
        RankedTensorType::get({1}, scalarLoad.getOther().getType(), blockedEnc);
    auto newOther = tt::SplatOp::create(builder, op->getLoc(), newOtherTy,
                                        scalarLoad.getOther());
    scalarLoad.getOtherMutable().assign(newOther);
  }
  auto newDstTy = RankedTensorType::get({1}, scalarTy, blockedEnc);
  scalarLoad.getResult().setType(newDstTy);
  builder.setInsertionPointAfter(op);
  Operation *firstUse = getFirstUseOfPipelinedOp({op}, forOp, schedule);
  builder.setStageCluster(schedule[firstUse]);
  Operation *unsplat = tt::UnsplatOp::create(builder, op->getLoc(), scalarTy,
                                             scalarLoad.getResult());
  scalarLoad.getResult().replaceAllUsesExcept(unsplat->getResult(0), unsplat);
}

static void
createMUSATMABarrierAndWait(scf::ForOp forOp,
                            llvm::MapVector<Operation *, AsyncLoad> &asyncLoads,
                            llvm::MapVector<int, LoadGroupInfo> &loadGroups,
                            tt::CoarseSchedule &schedule) {
  OpBuilder initBuilder(forOp);
  initBuilder.setInsertionPoint(forOp);
  Location loopLoc = forOp.getLoc();
  Value zeroPhase = arith::ConstantIntOp::create(initBuilder, loopLoc, 0, 32);
  Value arriveCount = arith::ConstantIntOp::create(initBuilder, loopLoc, 1, 32);

  SmallVector<SmallVector<Operation *>> commonWaitGroups;
  llvm::SmallDenseSet<Operation *> visited;
  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    if (!tt::isTMALoad(loadOp) || visited.count(loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<Operation *> group;
    Block *loadBlock = loadOp->getBlock();
    auto addToGroup = [&](Operation *groupLoadOp) {
      group.push_back(groupLoadOp);
      visited.insert(groupLoadOp);
      bool sharedFirst = !mustLoadToRegisters(groupLoadOp);
      for (Operation *user : groupLoadOp->getUsers()) {
        if (sharedFirst) {
          auto alloc = dyn_cast<ttg::LocalAllocOp>(user);
          if (alloc && alloc->getBlock() == loadBlock) {
            SmallVector<Operation *> nonViewUsers;
            collectNonViewUsers(alloc, nonViewUsers);
            for (Operation *nonViewUser : nonViewUsers) {
              Operation *userInBlock =
                  loadBlock->findAncestorOpInBlock(*nonViewUser);
              if (userInBlock)
                users.insert(userInBlock);
            }
            continue;
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(loadOp);
    Operation *nextOp = loadOp->getNextNode();
    int numBuffers = asyncLoad.stageDiff;
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (tt::isTMALoad(nextOp) && asyncLoads.count(nextOp)) {
        if (asyncLoads[nextOp].stageDiff != numBuffers)
          break;
        if (group.size() > 0 && schedule[group[0]] == schedule[nextOp]) {
          addToGroup(nextOp);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    commonWaitGroups.push_back(group);
  }

  for (SmallVector<Operation *> &group : commonWaitGroups) {
    int64_t sizeInBytes = 0;
    int numBuffers = asyncLoads[group[0]].stageDiff;
    auto reserved = triton::musa::reserveBarrierIdRange(forOp, numBuffers);
    if (failed(reserved)) {
      forOp.emitOpError("unable to reserve MUSA async barrier ids for "
                        "pipelined TME load group");
      llvm::report_fatal_error("Fatal pipeliner error");
    }
    int32_t barrierBase = *reserved;
    for (int32_t slot = 0; slot < numBuffers; ++slot) {
      Value barId = arith::ConstantIntOp::create(initBuilder, loopLoc,
                                                 barrierBase + slot, 32);
      triton::musa::InitArrivalOp::create(initBuilder, loopLoc, barId,
                                          arriveCount, zeroPhase);
    }
    LoadGroupInfo &loadGroup = loadGroups.find(numBuffers)->second;
    for (Operation *op : group) {
      auto tensorTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      int64_t loadSize = product(ttg::getShapePerCTA(tensorTy));
      sizeInBytes += loadSize * tensorTy.getElementTypeBitWidth() / 8;
    }

    tt::OpBuilderForStage builder(forOp.getLoc(), group[0], schedule);
    builder.setInsertionPoint(group[0]);
    builder.setStageCluster(schedule[group[0]]);
    Value issueBarId = materializeBarrierIdValue(
        builder, group[0]->getLoc(), barrierBase, loadGroup.insertIdx);
    Value transBytes = arith::ConstantIntOp::create(builder, group[0]->getLoc(),
                                                    sizeInBytes, 32);
    Value pred =
        arith::ConstantIntOp::create(builder, group[0]->getLoc(), 1, 1);
    auto addTrans = triton::musa::BarrierAddTransOp::create(
        builder, group[0]->getLoc(), issueBarId, transBytes, pred);

    builder.setInsertionPointAfter(group.back());
    builder.setStageCluster(schedule[group.back()]);
    auto arrive = triton::musa::ArriveBarrierNoRetOp::create(
        builder, group.back()->getLoc(), issueBarId, pred);

    Operation *firstUse = getFirstUseOfPipelinedOp(group, forOp, schedule);
    builder.setInsertionPointAfter(arrive);
    builder.setStageCluster(schedule[firstUse]);
    Value waitBarId = materializeBarrierIdValue(
        builder, firstUse->getLoc(), barrierBase, loadGroup.extractIdx);
    Value waitPhase =
        loadGroup.yieldPhase ? loadGroup.yieldPhase : loadGroup.phase;
    auto wait = triton::musa::WaitBarrierOp::create(builder, firstUse->getLoc(),
                                                    waitBarId, waitPhase);

    for (Operation *op : group) {
      asyncLoads[op].barrier = issueBarId;
      asyncLoads[op].waitOp = wait;
    }
  }
}

static bool loadRequiresAdditionalBuffer(Operation *loadOp) {
  auto isMusaTarget = [&]() {
    auto module = loadOp->getParentOfType<ModuleOp>();
    if (!module)
      return false;
    auto targetAttr = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    return targetAttr && targetAttr.getValue().starts_with("musa:");
  };

  std::function<void(Value, SmallVector<Operation *> &)> collectNonViewUsers =
      [&](Value value, SmallVector<Operation *> &out) {
        for (Operation *user : value.getUsers()) {
          if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
            for (Value result : user->getResults())
              if (isa<ttg::MemDescType>(result.getType()))
                collectNonViewUsers(result, out);
          } else {
            out.push_back(user);
          }
        }
      };
  std::function<bool(Operation *, DenseSet<Operation *> &)> hasDotConsumer =
      [&](Operation *op, DenseSet<Operation *> &visited) -> bool {
    if (!visited.insert(op).second)
      return false;
    if (isa<tt::DotOp>(op))
      return true;
    for (Operation *user : op->getUsers()) {
      if (hasDotConsumer(user, visited))
        return true;
    }
    return false;
  };
  if (!mustLoadToRegisters(loadOp)) {
    SmallVector<Value> landingMemDescs;
    if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(loadOp)) {
      if (failed(
              triton::musa::resolveDescriptorLoadLandingMemDescType(descLoad)))
        return false;
      landingMemDescs =
          triton::musa::collectCanonicalLandingMemDescRoots(descLoad);
    } else if (loadOp->hasOneUse()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin()))
        landingMemDescs.push_back(alloc.getResult());
    }
    for (Value memDesc : landingMemDescs) {
      SmallVector<Operation *> nonViewUsers;
      collectNonViewUsers(memDesc, nonViewUsers);
      if (llvm::any_of(nonViewUsers, [&](Operation *op) {
            if (isa<ttng::WarpGroupDotOp>(op))
              return true;
            if (isMusaTarget()) {
              DenseSet<Operation *> visited;
              return hasDotConsumer(op, visited);
            }
            return false;
          })) {
        return true;
      }
    }
  }
  return false;
}

static FailureOr<scf::ForOp>
lowerLoads(scf::ForOp forOp, tt::CoarseSchedule &schedule,
           triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  auto module = forOp->getParentOfType<ModuleOp>();
  auto targetAttr = module
                        ? module->getAttrOfType<StringAttr>(ttg::AttrTargetName)
                        : StringAttr();
  bool isMusaTarget = targetAttr && targetAttr.getValue().starts_with("musa:");
  auto hasDotConsumer = [&](Operation *loadOp) {
    DenseSet<Operation *> visited;
    std::function<bool(Operation *)> dfs = [&](Operation *op) -> bool {
      if (!visited.insert(op).second)
        return false;
      if (isa<tt::DotOpInterface>(op))
        return true;
      for (Operation *user : op->getUsers()) {
        if (dfs(user))
          return true;
      }
      return false;
    };
    for (Value result : loadOp->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (dfs(user))
          return true;
      }
    }
    return false;
  };

  llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  llvm::SmallVector<Operation *> scalarLoads;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      continue;

    if (isa<tt::DescriptorGatherOp>(op)) {
      op.emitOpError("pipelined descriptor_gather is not supported on MUSA");
      return failure();
    }

    int stageDiff = getDefUseStageDiff(&op, forOp, schedule);
    if (stageDiff == 0)
      continue;

    ttg::SharedEncodingTrait sharedEncoding;
    ttg::MemDescType landingMemDescTy;
    bool canUseAsyncCp = false;
    int contiguity = 1;
    if (!isa<RankedTensorType>(op.getResultTypes()[0])) {
      canUseAsyncCp = op.getResultTypes()[0].getIntOrFloatBitWidth() >= 32;
      auto numCTAs = ttg::lookupNumCTAs(forOp);
      sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
          forOp.getContext(), 1, 1, 1, {0},
          ttg::CGAEncodingAttr::get1DLayout(forOp.getContext(), numCTAs));
      auto sharedSpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
      landingMemDescTy = ttg::MemDescType::get(
          ArrayRef<int64_t>({1}), op.getResultTypes()[0], sharedEncoding,
          sharedSpace, /*mutableMemory=*/true);
      if (canUseAsyncCp)
        scalarLoads.push_back(&op);
    } else {
      if (isMusaTarget && isa<tt::DescriptorLoadOp>(op)) {
        auto resolvedLandingTy =
            triton::musa::resolveDescriptorLoadLandingMemDescType(
                cast<tt::DescriptorLoadOp>(&op));
        if (failed(resolvedLandingTy)) {
          op.emitOpError("pipelined descriptor load requires normalized "
                         "canonical landing memdesc encoding");
          return failure();
        }
        landingMemDescTy = *resolvedLandingTy;
        sharedEncoding =
            dyn_cast<ttg::SharedEncodingTrait>(landingMemDescTy.getEncoding());
      } else {
        sharedEncoding = tt::getSharedEncoding(&op);
        landingMemDescTy = getGenericMusaLoadMemDescType(&op, sharedEncoding);
      }
      canUseAsyncCp =
          isa<tt::LoadOp>(op) &&
          canBeConvertedToAsyncLoad(cast<tt::LoadOp>(op), axisInfoAnalysis);
      int copyVecBytes = tt::getCopyVecBytes(
          cast<RankedTensorType>(op.getResultTypes()[0]), sharedEncoding);
      canUseAsyncCp &= copyVecBytes >= 4;
      if (canUseAsyncCp) {
        auto loadOp = cast<tt::LoadOp>(op);
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));
        contiguity = vec;
      }
    }

    if (canUseAsyncCp || tt::isTMALoad(&op)) {
      if (loadRequiresAdditionalBuffer(&op))
        stageDiff += 1;
      if (isMusaTarget && hasDotConsumer(&op)) {
        if (tt::isTMALoad(&op))
          stageDiff += 1;
        if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(&op)) {
          if (auto sqmmaLandingTy =
                  triton::musa::getUniqueCanonicalLandingSqmmaMemDescType(
                      descLoad)) {
            landingMemDescTy = *sqmmaLandingTy;
            sharedEncoding = dyn_cast<ttg::SharedEncodingTrait>(
                landingMemDescTy.getEncoding());
          }
        }
        if (isa<tt::LoadOp>(op)) {
          if (auto sqmmaLandingTy =
                  getSqmmaOperandLandingMemDescType(cast<tt::LoadOp>(&op))) {
            stageDiff += 1;
            landingMemDescTy = sqmmaLandingTy;
            sharedEncoding = dyn_cast<ttg::SharedEncodingTrait>(
                landingMemDescTy.getEncoding());
          }
        }
      }
      auto &asyncLoad = asyncLoads[&op];
      asyncLoad.stageDiff = stageDiff;
      asyncLoad.contiguity = contiguity;
      asyncLoad.landingMemDescTy = landingMemDescTy;
    } else if (stageDiff > 1) {
      op.emitRemark() << "Pipelining load that cannot use vectorized copy. "
                         "This will likely lead to pipelining in registers and "
                         "severe performance degradation.";
    }
  }
  for (Operation *op : scalarLoads)
    convertScalarToTensorLoad(op, schedule, forOp);

  if (asyncLoads.empty())
    return forOp;

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    Value alloc = createAlloc(forOp, loadOp, asyncLoad.landingMemDescTy,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    loadGroups.insert({asyncLoad.stageDiff, {}});
    if (tt::isTMALoad(loadOp))
      loadGroups[asyncLoad.stageDiff].hasTMALoad = true;
  }
  IRRewriter builder(forOp);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [numBuffers, loadGroup] : loadGroups) {
    Value initCounter = minusOne;
    newOperands.push_back(initCounter);
    newOperands.push_back(initCounter);
    if (loadGroup.hasTMALoad)
      newOperands.push_back(zero);
  }

  forOp = addIterArgsToLoop(builder, forOp, newOperands);

  auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i)
    forYield.getResultsMutable().append(newOperands[i]);

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  auto findGroupIssueAnchor = [&](int numBuffers) -> Operation * {
    for (auto &[op, asyncLoad] : asyncLoads)
      if (asyncLoad.stageDiff == numBuffers)
        return op;
    return nullptr;
  };
  auto findGroupConsumerAnchor = [&](int numBuffers) -> Operation * {
    SmallVector<Operation *> group;
    for (auto &[op, asyncLoad] : asyncLoads)
      if (asyncLoad.stageDiff == numBuffers)
        group.push_back(op);
    if (group.empty())
      return nullptr;
    return getFirstUseOfPipelinedOp(group, forOp, schedule);
  };
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    Value insertIdx = forOp.getBody()->getArgument(argIdx++);
    Value extractIdx = forOp.getBody()->getArgument(argIdx++);
    Value phase;
    if (loadGroup.hasTMALoad)
      phase = forOp.getBody()->getArgument(argIdx++);
    loadGroup.phase = phase;

    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    Value numBuffersVal =
        arith::ConstantIntOp::create(builder, loc, numBuffers, 32);
    loadGroup.insertIdx = tt::createIncrementModulo(builder, loc, insertIdx,
                                                    numBuffersVal, zero, one);
    Value cndExt;
    loadGroup.extractIdx = tt::createIncrementModulo(
        builder, loc, extractIdx, numBuffersVal, zero, one, &cndExt);
    if (phase) {
      Value nextPhase = arith::XOrIOp::create(builder, loc, phase, one);
      loadGroup.yieldPhase =
          arith::SelectOp::create(builder, loc, cndExt, nextPhase, phase);
    }

    (void)findGroupIssueAnchor(numBuffers);
    Operation *consumerAnchor = findGroupConsumerAnchor(numBuffers);
    if (consumerAnchor && loadGroup.yieldPhase) {
      auto [consumerStage, consumerCluster] = schedule[consumerAnchor];
      scheduleScalarPipelineValue(schedule, loadGroup.yieldPhase, consumerStage,
                                  consumerCluster);
    }
  }

  createMUSATMABarrierAndWait(forOp, asyncLoads, loadGroups, schedule);

  bool hasAsyncLoads = false;
  for (auto &[op, asyncLoad] : asyncLoads) {
    LoadGroupInfo &loadGroup = loadGroups[asyncLoad.stageDiff];
    Value insertIdx = loadGroup.insertIdx;
    Value extractIdx = loadGroup.extractIdx;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      asyncLoad.contiguity, schedule);
      hasAsyncLoads = true;
    } else if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
      tt::OpBuilderForStage copyBuilder(loadOp.getLoc(), forOp, schedule);
      copyBuilder.setInsertionPoint(loadOp);
      copyBuilder.setStageCluster(schedule[loadOp]);
      auto view =
          createMusaSingleBufferView(copyBuilder, asyncLoad.alloc, insertIdx);
      if (auto allocOp = asyncLoad.alloc.getDefiningOp())
        triton::musa::copyCanonicalLandingSqmmaAttrs(loadOp, allocOp);
      if (auto viewOp = view.getDefiningOp())
        triton::musa::copyCanonicalLandingSqmmaAttrs(loadOp, viewOp);
      Value pred =
          arith::ConstantIntOp::create(copyBuilder, loadOp.getLoc(), 1, 1);
      auto blockTy = loadOp.getDesc().getType().getSignlessBlockType();
      auto coord = triton::musa::materializeTMECoordValues(
          loadOp.getLoc(), loadOp.getIndices(), copyBuilder);
      if (failed(coord)) {
        loadOp.emitOpError("unable to materialize pipelined TME block info");
        return failure();
      }
      auto config = triton::musa::resolveFinalTMECopyConfig(
          cast<ttg::MemDescType>(view.getType()), blockTy.getShape(),
          triton::musa::TMECopyKind::GlobalToLocal);
      if (failed(config)) {
        loadOp.emitOpError("unable to resolve pipelined TME load config");
        return failure();
      }
      triton::musa::createAsyncTMECopyGlobalToLocal(
          copyBuilder, loadOp.getLoc(), loadOp.getDesc(), *coord,
          asyncLoad.barrier, view, pred, *config);

      copyBuilder.setInsertionPointAfter(asyncLoad.waitOp);
      copyBuilder.setStageCluster(
          schedule[getFirstUseOfPipelinedOp({loadOp}, forOp, schedule)]);
      auto viewLoad =
          createMusaSingleBufferView(copyBuilder, asyncLoad.alloc, extractIdx);
      if (auto viewLoadOp = viewLoad.getDefiningOp())
        triton::musa::copyCanonicalLandingSqmmaAttrs(loadOp, viewLoadOp);
      IRRewriter rewriter(loadOp.getContext(), &copyBuilder);
      rewriter.setInsertionPoint(copyBuilder.getInsertionBlock(),
                                 copyBuilder.getInsertionPoint());
      replaceDescriptorUsesWithMemDescOrLocalLoad(loadOp, viewLoad, rewriter);
      schedule.erase(loadOp);
      loadOp->erase();
    }
  }

  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
    if (loadGroup.phase)
      forYield.setOperand(argIdx++, loadGroup.yieldPhase ? loadGroup.yieldPhase
                                                         : loadGroup.phase);
  }

  if (failed(validateMusaScheduleClusters(forOp, schedule))) {
    forOp.emitOpError("invalid cluster mapping before scheduling MUSA pipeline "
                      "dependencies");
    return failure();
  }
  scheduleDependencies(forOp, schedule);

  if (hasAsyncLoads) {
    builder.setInsertionPointAfter(forOp);
    ttg::AsyncWaitOp::create(builder, loc, ValueRange({}), 0);
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!schedule.count(&op)) {
      op.emitError() << "op not found in the schedule";
      return failure();
    }
  }
  return forOp;
}

static LogicalResult musaLowerLoops(ModuleOp moduleOp, int defaultNumStages) {
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  auto lowerLoopWithSchedule =
      [&](scf::ForOp forOp, tt::CoarseSchedule &schedule) -> LogicalResult {
    auto lowered = lowerLoads(forOp, schedule, axisInfoAnalysis);
    if (failed(lowered))
      return failure();
    scf::ForOp newForOp =
        triton::musa::pipeline::lowerTMADescriptors(*lowered, schedule);
    schedule.serialize(newForOp);
    return success();
  };

  for (scf::ForOp forOp : loops) {
    tt::CoarseSchedule schedule;
    if (succeeded(schedule.deSerialize(forOp))) {
      if (failed(lowerLoopWithSchedule(forOp, schedule)))
        return failure();
      continue;
    }

    {
      auto synthesized = synthesizeMusaSqmmaSchedule(forOp, defaultNumStages);
      if (failed(synthesized))
        continue;
      if (failed(lowerLoopWithSchedule(forOp, *synthesized)))
        return failure();
    }
  }
  return success();
}

static void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      continue;

    std::vector<std::pair<Operation *, unsigned>> finalSchedule =
        schedule.createFinalSchedule(forOp);
    tt::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = false;
    options.predicateFn = musaWrapInMaskOp;
    options.getScheduleFn =
        [&](scf::ForOp,
            std::vector<std::pair<Operation *, unsigned>> &loopSchedule) {
          loopSchedule = finalSchedule;
        };

    bool keepPredicateStage = forOp->hasAttr("__test_keep_predicate_stage");
    if (keepPredicateStage) {
      options.emitPredicateStageFn = [](RewriterBase &rewriter,
                                        Value inductionVar, Value upperBound,
                                        Value step, uint64_t maxStage,
                                        uint64_t stage) {
        return ttg::PredicateStageOp::create(rewriter, inductionVar.getLoc(),
                                             inductionVar, upperBound, step,
                                             maxStage, stage);
      };
    }

    IRRewriter rewriter(forOp);
    if (failed(tt::pipelineForLoop(rewriter, forOp, options)))
      continue;
  }

  assert(moduleOp.getOps<ttg::PredicateStageOp>().empty() &&
         "PredicateStageOp should be resolved after the pipeline expansion");
  assert(verify(moduleOp).succeeded());
  musaResolveMaskOp(moduleOp);
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUPIPELINE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUPipelinePass
    : impl::TritonMUSAGPUPipelineBase<TritonMUSAGPUPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (failed(musaLowerLoops(moduleOp, numStages)))
      return signalPassFailure();
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// TritonMUSAGPUPipeline internal IR Dump After: "
             "LowerLoops\n"
          << moduleOp << "\n\n\n";
    }

    expandLoops(moduleOp);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// TritonMUSAGPUPipeline internal IR Dump After: "
             "ExpandLoops\n"
          << moduleOp << "\n\n\n";
    }

    tt::removePipeliningAttributes(moduleOp);
    triton::musa::pipeline::pipelineSqmma(moduleOp, numStages);
    tt::updateWaits(moduleOp);

    auto *arithDialect =
        moduleOp.getContext()->getLoadedDialect<arith::ArithDialect>();
    RewritePatternSet patterns(moduleOp.getContext());
    arithDialect->getCanonicalizationPatterns(patterns);
    if (applyPatternsGreedily(moduleOp, std::move(patterns)).failed())
      return signalPassFailure();

    SmallVector<scf::ForOp> loops;
    moduleOp.walk([&](scf::ForOp forOp) {
      if (tt::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });
    for (scf::ForOp forOp : loops) {
      auto pipelined = triton::musa::pipeline::pipelineTMEStores(forOp);
      if (failed(pipelined))
        return signalPassFailure();
    }
  }
};

} // namespace mlir
