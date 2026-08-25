#include "Utility.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <functional>
#include <map>
#include <optional>
#include <set>

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#ifdef __TLE__
#include "tle/dialect/include/Analysis/TlePipeEffectAnalysis.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#endif
#include "llvm/ADT/STLExtras.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
#ifdef __TLE__
namespace ttle = mlir::triton::tle;
#endif
namespace mlir {

#define DEBUG_TYPE "tritongpu-warp-spec-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#ifdef __TLE__
constexpr llvm::StringLiteral
    kTleParticipantConsumerReleaseAttr("tle.participant_consumer_release");
#endif

// Lower to use GetCanonicalWarpIdOp.
// In Hopper, each task is a warpgroup consisting of 4 warps.
static const int WARPS_PER_TASK = 4;
static const int THREADS_PER_TASK = 128;

Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool emptyBarrier) {
  auto loc = op->getLoc();
  assert(isa<ttnvws::ProducerAcquireOp>(op) || isa<ttnvws::ConsumerWaitOp>(op));
  Value curPhase;
  if (auto acq = dyn_cast<ttnvws::ProducerAcquireOp>(op))
    curPhase = acq.getPhase();
  else if (auto wait = dyn_cast<ttnvws::ConsumerWaitOp>(op))
    curPhase = wait.getPhase();
  if (emptyBarrier) {
    // curPhase = curPhase xor True for emptyBarrier.
    Value _1_1b = arith::ConstantIntOp::create(builder, loc, 1, 1);
    curPhase = mlir::arith::XOrIOp::create(builder, loc, curPhase, _1_1b);
  }
  LLVM_DEBUG(curPhase.dump());
  return curPhase;
}

void processProducerAcquireOp(OpBuilder &builder, ttnvws::ProducerAcquireOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, true);
  auto i32Ty = builder.getIntegerType(32);
  phase = arith::ExtUIOp::create(builder, loc, i32Ty, phase);
  auto waitOp = ttng::WaitBarrierOp::create(builder, loc, bufferEmpty, phase);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processProducerCommitOp(OpBuilder &builder, ttnvws::ProducerCommitOp op,
                             Value bufferFull, ttnvws::TokenLoadType loadType,
                             unsigned fullCnt) {
  auto loc = op.getLoc();
  Operation *arriveOp = nullptr;

  if (op.getCommitKind() ==
      ttnvws::ProducerCommitKind::AsyncCopyMbarrierArrive) {
    arriveOp = ttng::AsyncCopyMbarrierArriveOp::create(builder, loc, bufferFull,
                                                       /*noIncrement=*/true);
  } else if (loadType == ttnvws::TokenLoadType::TMALoadOp ||
             loadType == ttnvws::TokenLoadType::LocalStoreOp) {
    // Get the count from the barriers: trace the local_alloc for the barrier
    // then find the count from init_barrier
#ifdef __TLE__
    unsigned arriveCnt = fullCnt;
    if (auto attr = op->getAttrOfType<IntegerAttr>("arrive_count"))
      arriveCnt = static_cast<unsigned>(attr.getInt());
    auto arriveBarrier =
        ttng::ArriveBarrierOp::create(builder, loc, bufferFull, arriveCnt);
    // Local-store pipe commits publish ordinary shared-memory writes from all
    // producer threads, while mbarrier.arrive is executed only by the elected
    // thread. Add an explicit CTA release fence so consumer waits observe the
    // producer partition's shared-memory stores.
    if (loadType == ttnvws::TokenLoadType::LocalStoreOp)
      arriveBarrier.setReleaseFence(true);
    // For proven local-store participants, each writer lane performs its own
    // release fence and single arrive. This removes the full partition
    // rendezvous while preserving the publish-before-ready contract: a lane can
    // only arrive after its own stores, and the barrier completes only after
    // all inferred writer participants have arrived.
    if (op.getCommitKind() ==
        ttnvws::ProducerCommitKind::ParticipantBarrierArrive)
      arriveBarrier.setParticipantArrive(true);
    arriveOp = arriveBarrier;
#else
    arriveOp = ttng::ArriveBarrierOp::create(builder, loc, bufferFull, fullCnt);
#endif
  } else {
    assert(false);
  }

  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

static int getTMACopyLoadSize(ttg::TMACopyOp copy) {
  auto dstTy = cast<ttg::MemDescType>(copy.getDst().getType());
  auto shapePerCTA = ttg::getShapePerCTA(dstTy.getEncoding(), dstTy.getShape());
  return product(shapePerCTA) * dstTy.getElementType().getIntOrFloatBitWidth() /
         8;
}

static bool canInterleaveBeforeTmaCopyCommit(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  return isMemoryEffectFree(op);
}

static SmallVector<ttg::TMACopyOp>
collectTmaCopiesForCommit(ttnvws::ProducerCommitOp op) {
  SmallVector<ttg::TMACopyOp> copies;
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      copies.push_back(tmaCopy);
      continue;
    }
    if (canInterleaveBeforeTmaCopyCommit(prev))
      continue;
    break;
  }
  std::reverse(copies.begin(), copies.end());
  return copies;
}

#ifdef __TLE__
static FailureOr<SmallVector<ttg::TMACopyOp>>
collectTmaCopiesForCommitTle(ttnvws::ProducerCommitOp op);
#endif

static LogicalResult processProducerCommitTmaCopyOp(OpBuilder &builder,
                                                    ttnvws::ProducerCommitOp op,
                                                    Value bufferFull) {
#ifdef __TLE__
  FailureOr<SmallVector<ttg::TMACopyOp>> maybeTmaCopies =
      collectTmaCopiesForCommitTle(op);
  if (failed(maybeTmaCopies))
    return failure();
  SmallVector<ttg::TMACopyOp> tmaCopies = std::move(*maybeTmaCopies);
#else
  SmallVector<ttg::TMACopyOp> tmaCopies = collectTmaCopiesForCommit(op);
#endif
  if (tmaCopies.empty())
    return op.emitOpError("with tma_copy_barrier_arrive must be preceded by "
                          "at least one ttg.tma_copy");

  Operation *bufferFullDef = bufferFull.getDefiningOp();
  if (bufferFullDef &&
      bufferFullDef->getBlock() == tmaCopies.front()->getBlock())
    bufferFullDef->moveBefore(tmaCopies.front());

  int sizeInBytes = 0;
  for (ttg::TMACopyOp copy : tmaCopies) {
    if (!isa<tt::TensorDescType>(copy.getSrc().getType()) ||
        !isa<ttg::MemDescType>(copy.getDst().getType())) {
      return copy.emitOpError("used by tma_copy_barrier_arrive must copy from "
                              "a tensor descriptor to a memdesc");
    }
    sizeInBytes += getTMACopyLoadSize(copy);
  }

  SmallVector<AsyncTaskId> taskIds = getAsyncTaskIds(op.getOperation());
  builder.setInsertionPoint(tmaCopies.front());
  auto loc = tmaCopies.front().getLoc();
  auto pred = arith::ConstantIntOp::create(builder, loc, 1, 1);
  setAsyncTaskIds(pred, taskIds);
  auto expect = ttng::BarrierExpectOp::create(builder, loc, bufferFull,
                                              sizeInBytes, pred);
  setAsyncTaskIds(expect, taskIds);

  for (ttg::TMACopyOp copy : tmaCopies) {
    auto srcTy = cast<tt::TensorDescType>(copy.getSrc().getType());
    auto indices = ttng::translateTMAIndices(builder, copy.getLoc(),
                                             srcTy.getBlockType().getEncoding(),
                                             copy.getIndices());
    builder.setInsertionPoint(copy);
    auto lowered = ttng::AsyncTMACopyGlobalToLocalOp::create(
        builder, copy.getLoc(), copy.getSrc(), indices, bufferFull,
        copy.getDst(), pred);
    lowered.setEvict(copy.getEvict());
    setAsyncTaskIds(lowered, taskIds);
    copy.erase();
  }

  return success();
}

void processConsumerWaitOp(OpBuilder &builder, ttnvws::ConsumerWaitOp op,
                           Value bufferFull) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, false);
  auto i32Ty = builder.getIntegerType(32);
  phase = arith::ExtUIOp::create(builder, loc, i32Ty, phase);
  auto waitOp = ttng::WaitBarrierOp::create(builder, loc, bufferFull, phase);
#ifdef __TLE__
  if (op.getWaitMode() == ttnvws::ConsumerWaitMode::ParticipantCounted)
    waitOp.setWaitMode(ttng::BarrierWaitMode::ParticipantCounted);
#endif
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processConsumerReleaseOp(OpBuilder &builder, ttnvws::ConsumerReleaseOp op,
                              Value bufferEmpty, int numCTAs,
                              unsigned releaseCnt) {
  auto loc = op.getLoc();
#ifdef __TLE__
  SmallVector<Value> releasedFields(op.getReleasedFields());
  auto arriveOp = ttng::ArriveBarrierOp::create(
      builder, loc, bufferEmpty, releaseCnt, op.getIdx(), releasedFields);
  // TLE pipe lowering marks releases whose reader task and released payload
  // contract has been established. In that case each reader lane can arrive
  // once after its own reads. Generic NVWS consumer releases keep the
  // conservative elected arrive path.
  if (op->hasAttr(kTleParticipantConsumerReleaseAttr) &&
      !releasedFields.empty())
    arriveOp.setParticipantArrive(true);
#else
  auto arriveOp =
      ttng::ArriveBarrierOp::create(builder, loc, bufferEmpty, releaseCnt);
#endif
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

static std::optional<unsigned> getTokenCountOverride(ttnvws::CreateTokenOp op,
                                                     StringRef attrName) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

#ifdef __TLE__
constexpr llvm::StringLiteral
    kTleInferArriveCountAttr("tle.infer_arrive_count");
constexpr llvm::StringLiteral
    kTleInferFullCountOffsetAttr("tle.infer_full_count_offset");

static Value getMemDescRoot(Value value) {
  Value current = value;
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = index.getSrc();
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = subslice.getSrc();
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = reinterpret.getSrc();
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = trans.getSrc();
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = reshape.getSrc();
      continue;
    }
    break;
  }
  return current;
}

static bool canInterleaveBeforeInferredParticipantCommit(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (ttle::isCtaInvariantSpecialRegisterRead(op))
    return true;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && ttle::isNonSharedPointer(load.getPtr());
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    if (ttle::getLocalStoreTarget(op))
      return true;
    return ttle::isNonSharedPointer(store.getPtr());
  }
  if (isa<ttg::LocalStoreOp, ttg::TMACopyOp>(op))
    return true;
  return false;
}

static bool canInterleaveBeforeTleMixedTmaCommit(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (ttle::isCtaInvariantSpecialRegisterRead(op))
    return true;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && ttle::isNonSharedPointer(load.getPtr());
  if (auto store = dyn_cast<tt::StoreOp>(op))
    return ttle::isNonSharedPointer(store.getPtr());
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return true;
  return false;
}

static bool isSameTokenSlot(ttnvws::ProducerCommitOp lhs,
                            ttnvws::ProducerCommitOp rhs) {
  return lhs.getToken() == rhs.getToken() &&
         ttle::sameIndexValue(lhs.getIdx(), rhs.getIdx());
}

static bool hasFollowingMixedPayloadCommit(ttnvws::ProducerCommitOp op) {
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (auto nextCommit = dyn_cast<ttnvws::ProducerCommitOp>(next))
      return isSameTokenSlot(op, nextCommit) &&
             (nextCommit.getCommitKind() ==
                  ttnvws::ProducerCommitKind::ParticipantBarrierArrive ||
              nextCommit.getCommitKind() ==
                  ttnvws::ProducerCommitKind::AsyncCopyMbarrierArrive);
    if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(next)) {
      if (arrive.getParticipantArrive())
        continue;
    }
    if (isa<ttng::AsyncCopyMbarrierArriveOp>(next))
      continue;
    if (canInterleaveBeforeTleMixedTmaCommit(next))
      continue;
    return false;
  }
  return false;
}

static LogicalResult
recordTleMixedLocalRoot(ttnvws::ProducerCommitOp commit, Value memdesc,
                        llvm::DenseSet<Value> &localRoots,
                        const llvm::DenseSet<Value> &tmaRoots) {
  Value root = getMemDescRoot(memdesc);
  if (tmaRoots.contains(root))
    return commit.emitOpError("cannot associate a mixed TMA/local-store "
                              "commit when a local-store payload aliases a "
                              "TMA payload memdesc root");
  localRoots.insert(root);
  return success();
}

static FailureOr<SmallVector<ttg::TMACopyOp>>
collectTmaCopiesForCommitTle(ttnvws::ProducerCommitOp op) {
  if (!hasFollowingMixedPayloadCommit(op))
    return collectTmaCopiesForCommit(op);

  SmallVector<ttg::TMACopyOp> copies;
  llvm::DenseSet<Value> tmaRoots;
  llvm::DenseSet<Value> localRoots;
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
    if (auto acquire = dyn_cast<ttnvws::ProducerAcquireOp>(prev)) {
      if (acquire.getToken() == op.getToken() &&
          ttle::sameIndexValue(acquire.getIdx(), op.getIdx()))
        break;
      return op.emitOpError("cannot associate mixed TMA copies across an "
                            "unrelated producer acquire");
    }
    if (isa<ttnvws::ProducerCommitOp>(prev))
      return op.emitOpError("cannot associate mixed TMA copies across an "
                            "unrelated producer commit");

    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      Value root = getMemDescRoot(tmaCopy.getDst());
      if (localRoots.contains(root))
        return op.emitOpError("cannot associate a mixed TMA/local-store "
                              "commit when a TMA payload aliases a "
                              "local-store payload memdesc root");
      tmaRoots.insert(root);
      copies.push_back(tmaCopy);
      continue;
    }

    if (auto target = ttle::getAsyncCopyTarget(prev)) {
      if (failed(recordTleMixedLocalRoot(op, target->memdesc, localRoots,
                                         tmaRoots)))
        return failure();
      continue;
    }

    if (auto target = ttle::getLocalStoreTarget(prev)) {
      if (failed(recordTleMixedLocalRoot(op, target->memdesc, localRoots,
                                         tmaRoots)))
        return failure();
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(prev)) {
      if (ttle::isSharedPointer(store.getPtr()))
        return op.emitOpError("cannot associate mixed TMA copies across an "
                              "opaque shared-memory store");
    }

    if (canInterleaveBeforeTleMixedTmaCommit(prev))
      continue;

    return op.emitOpError("cannot associate mixed TMA copies across ")
           << prev->getName();
  }

  std::reverse(copies.begin(), copies.end());
  return copies;
}

static FailureOr<unsigned> getTaskThreadCount(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitOpError("requires enclosing module to infer participant "
                           "barrier count");
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (numWarps <= 0 || threadsPerWarp <= 0)
    return op->emitOpError("requires positive num_warps and threads_per_warp "
                           "to infer participant barrier count");
  return static_cast<unsigned>(numWarps * threadsPerWarp);
}

static FailureOr<unsigned>
inferPrefixParticipantsFromLayout(ttnvws::ProducerCommitOp commit,
                                  Type valueType, unsigned taskThreadCount) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueType);
  if (!tensorTy || !tensorTy.hasStaticShape() || !tensorTy.getEncoding())
    return commit.emitOpError("requires encoded local-store value layout "
                              "before participant barrier count inference");

  int64_t numElements = tensorTy.getNumElements();
  if (numElements <= 0)
    return commit.emitOpError("requires non-empty local-store value shape for "
                              "participant barrier count inference");
  unsigned elemsPerThread = ttg::getTotalElemsPerThread(tensorTy);
  if (elemsPerThread == 0)
    return commit.emitOpError("requires positive elements-per-thread for "
                              "participant barrier count inference");

  int64_t participants = (numElements + elemsPerThread - 1) / elemsPerThread;
  if (participants <= 0)
    return commit.emitOpError("failed to infer participant barrier count");
  return static_cast<unsigned>(
      std::min<int64_t>(participants, taskThreadCount));
}

static FailureOr<unsigned>
inferParticipantArriveCount(ttnvws::ProducerCommitOp commit) {
  FailureOr<unsigned> taskThreadCount = getTaskThreadCount(commit);
  if (failed(taskThreadCount))
    return failure();

  std::optional<unsigned> participants;
  ttle::CompletedAsyncCopyState completedAsyncCopies;
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (auto acquire = dyn_cast<ttnvws::ProducerAcquireOp>(prev)) {
      if (acquire.getToken() == commit.getToken() &&
          ttle::sameIndexValue(acquire.getIdx(), commit.getIdx()))
        break;
      return commit.emitOpError("cannot infer participant barrier count across "
                                "an unrelated producer acquire");
    }
    if (auto prevCommit = dyn_cast<ttnvws::ProducerCommitOp>(prev)) {
      if (prevCommit.getToken() == commit.getToken() &&
          ttle::sameIndexValue(prevCommit.getIdx(), commit.getIdx()) &&
          prevCommit.getCommitKind() ==
              ttnvws::ProducerCommitKind::TmaCopyBarrierArrive)
        continue;
      return commit.emitOpError("cannot infer participant barrier count across "
                                "an unrelated producer commit");
    }
    if (auto wait = dyn_cast<ttg::AsyncWaitOp>(prev)) {
      ttle::recordCompletedAsyncWait(wait, completedAsyncCopies);
      continue;
    }
    if (auto asyncCommit = dyn_cast<ttg::AsyncCommitGroupOp>(prev)) {
      ttle::propagateCompletedAsyncCommitGroup(asyncCommit,
                                               completedAsyncCopies);
      continue;
    }
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(prev)) {
      auto target = ttle::getAsyncCopyTarget(prev);
      assert(target && "async copy must have a local destination");
      if (!ttle::isAsyncCopyComplete(asyncCopy, completedAsyncCopies))
        return commit.emitOpError("cannot infer participant barrier count "
                                  "from an async copy without a proven "
                                  "async_wait before the producer commit");
      FailureOr<unsigned> count = inferPrefixParticipantsFromLayout(
          commit, target->valueType, *taskThreadCount);
      if (failed(count))
        return failure();
      participants = participants ? std::max(*participants, *count) : *count;
      continue;
    }

    if (auto target = ttle::getLocalStoreTarget(prev)) {
      FailureOr<unsigned> count = inferPrefixParticipantsFromLayout(
          commit, target->valueType, *taskThreadCount);
      if (failed(count))
        return failure();
      participants = participants ? std::max(*participants, *count) : *count;
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(prev)) {
      if (ttle::isSharedPointer(store.getPtr()))
        return commit.emitOpError("cannot infer participant barrier count "
                                  "across an opaque shared-memory store");
    }

    if (!canInterleaveBeforeInferredParticipantCommit(prev))
      return commit.emitOpError(
                 "cannot infer participant barrier count across ")
             << prev->getName();
  }

  if (!participants)
    return commit.emitOpError("requires at least one local-store contributor "
                              "before inferred participant barrier commit");
  return *participants;
}

static LogicalResult
recordInferredParticipantCount(ttnvws::ProducerCommitOp commit,
                               std::optional<unsigned> &tokenCount) {
  if (!commit->hasAttr(kTleInferArriveCountAttr))
    return success();

  FailureOr<unsigned> count = inferParticipantArriveCount(commit);
  if (failed(count))
    return failure();

  if (tokenCount && *tokenCount != *count)
    return commit.emitOpError("infers participant barrier count ")
           << *count << " but the same token already inferred " << *tokenCount;
  tokenCount = *count;
  commit->setAttr(
      "arrive_count",
      IntegerAttr::get(IntegerType::get(commit.getContext(), 32), *count));
  commit->removeAttr(kTleInferArriveCountAttr);
  return success();
}
#endif

static bool isMBarrierInitSetupOp(Operation *op) {
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    return true;
  if (isa<ttng::InitBarrierOp>(op))
    return true;
  return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
         isMemoryEffectFree(op);
}

static bool closesMBarrierInitRun(gpu::BarrierOp barrier) {
  bool sawInit = false;
  for (Operation *op = barrier->getPrevNode(); op; op = op->getPrevNode()) {
    if (isa<gpu::BarrierOp>(op))
      break;
    if (!isMBarrierInitSetupOp(op))
      break;
    sawInit |= isa<ttng::InitBarrierOp>(op);
  }
  return sawInit;
}

static void coalesceMBarrierInitBarriersInBlock(Block &block) {
  gpu::BarrierOp pendingInitBarrier;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto barrier = dyn_cast<gpu::BarrierOp>(op)) {
      if (!closesMBarrierInitRun(barrier)) {
        pendingInitBarrier = {};
        continue;
      }
      if (pendingInitBarrier)
        pendingInitBarrier.erase();
      pendingInitBarrier = barrier;
      continue;
    }

    if (pendingInitBarrier && !isMBarrierInitSetupOp(&op))
      pendingInitBarrier = {};
  }
}

static void coalesceMBarrierInitBarriersInRegion(Region &region) {
  for (Block &block : region) {
    coalesceMBarrierInitBarriersInBlock(block);
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        coalesceMBarrierInitBarriersInRegion(nested);
  }
}

static void coalesceMBarrierInitBarriers(Operation *parentOp) {
  for (Region &region : parentOp->getRegions())
    coalesceMBarrierInitBarriersInRegion(region);
}

#ifdef __TLE__
enum TokenBarrierRequirement : unsigned {
  TokenNeedsNone = 0,
  TokenNeedsFull = 1,
  TokenNeedsEmpty = 2,
};

struct TokenBarrierValues {
  Value full;
  Value empty;
};

static unsigned getDirectTokenBarrierRequirement(Operation *user) {
  if (isa<ttnvws::ProducerCommitOp, ttnvws::ConsumerWaitOp>(user))
    return TokenNeedsFull;
  if (isa<ttnvws::ProducerAcquireOp, ttnvws::ConsumerReleaseOp>(user))
    return TokenNeedsEmpty;
  return TokenNeedsNone;
}

static FailureOr<tt::FuncOp> getTokenCallee(ModuleOp module,
                                             tt::CallOp call) {
  auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
  if (!callee)
    return call.emitOpError("cannot resolve token-bearing callee ")
           << call.getCallee();
  if (callee.isExternal() || !callee.getBody().hasOneBlock())
    return call.emitOpError("token-bearing callee must have one body block");
  return callee;
}

static FailureOr<unsigned> getTokenBarrierRequirement(
    Value token, ModuleOp module, DenseMap<Value, unsigned> &requirements,
    DenseSet<Value> &active) {
  if (auto it = requirements.find(token); it != requirements.end())
    return it->second;
  if (!active.insert(token).second)
    return emitError(token.getLoc())
           << "recursive token flow through device calls is unsupported";

  unsigned requirement = TokenNeedsNone;
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    unsigned direct = getDirectTokenBarrierRequirement(user);
    if (direct != TokenNeedsNone) {
      requirement |= direct;
      continue;
    }
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandNumber = use.getOperandNumber();
      if (operandNumber >= wsOp.getExplicitCaptures().size())
        return wsOp.emitOpError("token must be an explicit capture");
      for (Region *region : wsOp.getPartitionRegions()) {
        if (operandNumber >= region->getNumArguments())
          return wsOp.emitOpError("token capture has no partition argument");
        auto nested = getTokenBarrierRequirement(
            region->getArgument(operandNumber), module, requirements, active);
        if (failed(nested))
          return failure();
        requirement |= *nested;
      }
      continue;
    }
    if (auto call = dyn_cast<tt::CallOp>(user)) {
      auto callee = getTokenCallee(module, call);
      if (failed(callee))
        return failure();
      unsigned operandNumber = use.getOperandNumber();
      if (operandNumber >= callee->getNumArguments())
        return call.emitOpError("token operand has no callee argument");
      auto nested = getTokenBarrierRequirement(
          callee->getArgument(operandNumber), module, requirements, active);
      if (failed(nested))
        return failure();
      requirement |= *nested;
      continue;
    }
    return user->emitOpError("unexpected use of an NVWS token");
  }

  active.erase(token);
  requirements[token] = requirement;
  return requirement;
}

static LogicalResult collectTokenLifecycleUsers(
    Value token, ModuleOp module, DenseSet<Value> &visited,
    DenseSet<Operation *> &lifecycleUsers) {
  if (!visited.insert(token).second)
    return success();
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    if (getDirectTokenBarrierRequirement(user) != TokenNeedsNone) {
      lifecycleUsers.insert(user);
      continue;
    }
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandNumber = use.getOperandNumber();
      for (Region *region : wsOp.getPartitionRegions()) {
        if (failed(collectTokenLifecycleUsers(
                region->getArgument(operandNumber), module, visited,
                lifecycleUsers)))
          return failure();
      }
      continue;
    }
    if (auto call = dyn_cast<tt::CallOp>(user)) {
      auto callee = getTokenCallee(module, call);
      if (failed(callee))
        return failure();
      if (failed(collectTokenLifecycleUsers(
              callee->getArgument(use.getOperandNumber()), module, visited,
              lifecycleUsers)))
        return failure();
      continue;
    }
    return user->emitOpError("unexpected use of an NVWS token");
  }
  return success();
}

static bool haveCompatibleTokenContracts(ttnvws::CreateTokenOp lhs,
                                         ttnvws::CreateTokenOp rhs) {
  if (lhs.getResult().getType() != rhs.getResult().getType() ||
      lhs.getLoadType() != rhs.getLoadType())
    return false;
  for (StringRef attr :
       ArrayRef<StringRef>{"full_count", "empty_count",
                           kTleInferFullCountOffsetAttr}) {
    if (lhs->getAttr(attr) != rhs->getAttr(attr))
      return false;
  }
  return true;
}

static void recordOriginalOperand(
    std::map<Operation *, SmallVector<unsigned>> &operands, Operation *op,
    unsigned operandNumber) {
  SmallVector<unsigned> &indices = operands[op];
  if (!llvm::is_contained(indices, operandNumber))
    indices.push_back(operandNumber);
}

static LogicalResult lowerTokenOperationsThroughCalls(Operation *parentOp,
                                                       int numCTAs) {
  auto module = dyn_cast<ModuleOp>(parentOp);
  if (!module)
    return parentOp->emitOpError(
        "interprocedural NVWS token lowering requires a module");

  SmallVector<ttnvws::CreateTokenOp> createTokens;
  module.walk([&](ttnvws::CreateTokenOp op) { createTokens.push_back(op); });
  if (createTokens.empty())
    return success();

  DenseMap<Value, unsigned> requirements;
  DenseSet<Value> active;
  DenseMap<Operation *, DenseSet<Operation *>> lifecycleUsersByToken;
  DenseSet<Operation *> allLifecycleUsers;
  SmallVector<ttnvws::CreateTokenOp> liveCreateTokens;
  for (ttnvws::CreateTokenOp create : createTokens) {
    auto requirement = getTokenBarrierRequirement(
        create.getResult(), module, requirements, active);
    if (failed(requirement))
      return failure();
    if (*requirement == TokenNeedsNone && create.getResult().use_empty()) {
      create.erase();
      continue;
    }
    if (*requirement == TokenNeedsNone)
      return create.emitOpError("has no reachable lifecycle users");
    liveCreateTokens.push_back(create);
    DenseSet<Value> visited;
    DenseSet<Operation *> users;
    if (failed(collectTokenLifecycleUsers(create.getResult(), module, visited,
                                          users)))
      return failure();
    allLifecycleUsers.insert(users.begin(), users.end());
    lifecycleUsersByToken.try_emplace(create.getOperation(), std::move(users));
  }
  createTokens = std::move(liveCreateTokens);
  if (createTokens.empty())
    return success();

  DenseMap<Operation *, unsigned> inferredParticipantCounts;
  for (Operation *user : allLifecycleUsers) {
    auto commit = dyn_cast<ttnvws::ProducerCommitOp>(user);
    if (!commit || !commit->hasAttr(kTleInferArriveCountAttr))
      continue;
    std::optional<unsigned> count;
    if (failed(recordInferredParticipantCount(commit, count)))
      return failure();
    assert(count && "inferred participant commit must produce a count");
    inferredParticipantCounts[user] = *count;
  }

  MLIRContext *context = module.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto barrierCTALayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);

  DenseMap<Value, TokenBarrierValues> barrierValues;
  DenseMap<Value, ttnvws::CreateTokenOp> tokenOrigins;
  DenseMap<Operation *, unsigned> tokenFullCounts;
  for (ttnvws::CreateTokenOp create : createTokens) {
    OpBuilder builder(create);
    Location loc = create.getLoc();
    unsigned requirement = requirements.lookup(create.getResult());
    Type arrayType = ttg::MemDescType::get(
        {create.getNumBuffers(), 1}, builder.getI64Type(), barrierEncoding,
        sharedMemorySpace, /*mutableMemory=*/true);
    Type slotType = ttg::MemDescType::get(
        {1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);

    TokenBarrierValues values;
    if (requirement & TokenNeedsFull)
      values.full = ttg::LocalAllocOp::create(builder, loc, arrayType, Value());
    if (requirement & TokenNeedsEmpty)
      values.empty =
          ttg::LocalAllocOp::create(builder, loc, arrayType, Value());
    barrierValues[create.getResult()] = values;
    tokenOrigins[create.getResult()] = create;

    unsigned fullCount = create.getLoadType() ==
                                 ttnvws::TokenLoadType::TMALoadOp
                             ? 1
                             : THREADS_PER_TASK;
    if (create.getLoadType() != ttnvws::TokenLoadType::TMALoadOp) {
      if (auto count = getTokenCountOverride(create, "full_count"))
        fullCount = *count;
    }
    if (auto offset =
            getTokenCountOverride(create, kTleInferFullCountOffsetAttr)) {
      std::optional<unsigned> inferred;
      for (Operation *user : lifecycleUsersByToken.lookup(create)) {
        auto it = inferredParticipantCounts.find(user);
        if (it == inferredParticipantCounts.end())
          continue;
        if (inferred && *inferred != it->second)
          return user->emitOpError("inconsistent inferred participant counts ")
                 << *inferred << " and " << it->second;
        inferred = it->second;
      }
      if (!inferred)
        return create.emitOpError(
            "requires an inferred participant producer commit for token "
            "full_count inference");
      fullCount = *inferred + *offset;
      create->removeAttr(kTleInferFullCountOffsetAttr);
    }
    tokenFullCounts[create.getOperation()] = fullCount;

    unsigned emptyCount = THREADS_PER_TASK;
    if (auto count = getTokenCountOverride(create, "empty_count"))
      emptyCount = *count;
    for (unsigned i = 0; i < create.getNumBuffers(); ++i) {
      Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
      if (values.full) {
        Value slot = ttg::MemDescIndexOp::create(builder, loc, slotType,
                                                 values.full, idx);
        ttng::InitBarrierOp::create(builder, loc, slot, fullCount);
      }
      if (values.empty) {
        Value slot = ttg::MemDescIndexOp::create(builder, loc, slotType,
                                                 values.empty, idx);
        ttng::InitBarrierOp::create(builder, loc, slot, emptyCount);
      }
    }
    assert(numCTAs == 1 && "remote CTA is not supported yet");
    if (values.full || values.empty)
      mlir::gpu::BarrierOp::create(builder, loc);
  }

  std::map<Operation *, SmallVector<unsigned>> callTokenOperands;
  std::map<Operation *, SmallVector<unsigned>> functionTokenArguments;
  std::map<Operation *, SmallVector<unsigned>> wsTokenOperands;
  DenseSet<Value> propagated;
  std::function<LogicalResult(Value)> propagate =
      [&](Value token) -> LogicalResult {
    if (!propagated.insert(token).second)
      return success();
    TokenBarrierValues values = barrierValues.lookup(token);
    ttnvws::CreateTokenOp origin = tokenOrigins.lookup(token);
    if (!origin)
      return emitError(token.getLoc()) << "token has no create-token origin";

    for (OpOperand &use : llvm::make_early_inc_range(token.getUses())) {
      Operation *user = use.getOwner();
      if (getDirectTokenBarrierRequirement(user) != TokenNeedsNone)
        continue;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned operandNumber = use.getOperandNumber();
        unsigned requirement = requirements.lookup(token);
        recordOriginalOperand(wsTokenOperands, wsOp, operandNumber);
        unsigned fullArgument = 0;
        unsigned emptyArgument = 0;
        if (requirement & TokenNeedsFull) {
          wsOp->insertOperands(wsOp.getNumOperands(), values.full);
          fullArgument = wsOp.getNumOperands() - 1;
          for (Region *region : wsOp.getPartitionRegions())
            region->addArgument(values.full.getType(), values.full.getLoc());
        }
        if (requirement & TokenNeedsEmpty) {
          wsOp->insertOperands(wsOp.getNumOperands(), values.empty);
          emptyArgument = wsOp.getNumOperands() - 1;
          for (Region *region : wsOp.getPartitionRegions())
            region->addArgument(values.empty.getType(), values.empty.getLoc());
        }
        for (Region *region : wsOp.getPartitionRegions()) {
          Value nested = region->getArgument(operandNumber);
          TokenBarrierValues nestedValues;
          if (requirement & TokenNeedsFull)
            nestedValues.full = region->getArgument(fullArgument);
          if (requirement & TokenNeedsEmpty)
            nestedValues.empty = region->getArgument(emptyArgument);
          barrierValues[nested] = nestedValues;
          tokenOrigins[nested] = origin;
          if (failed(propagate(nested)))
            return failure();
        }
        continue;
      }
      if (auto call = dyn_cast<tt::CallOp>(user)) {
        auto callee = getTokenCallee(module, call);
        if (failed(callee))
          return failure();
        unsigned operandNumber = use.getOperandNumber();
        Value calleeToken = callee->getArgument(operandNumber);
        unsigned requirement = requirements.lookup(calleeToken);
        TokenBarrierValues calleeValues = barrierValues.lookup(calleeToken);
        if (!calleeValues.full && !calleeValues.empty) {
          if (requirement & TokenNeedsFull) {
            unsigned index = callee->getNumArguments();
            if (failed(callee->insertArgument(index, values.full.getType(),
                                              DictionaryAttr{},
                                              calleeToken.getLoc())))
              return callee->emitOpError("cannot append full-barrier argument");
            calleeValues.full = callee->getArgument(index);
          }
          if (requirement & TokenNeedsEmpty) {
            unsigned index = callee->getNumArguments();
            if (failed(callee->insertArgument(index, values.empty.getType(),
                                              DictionaryAttr{},
                                              calleeToken.getLoc())))
              return callee->emitOpError(
                  "cannot append empty-barrier argument");
            calleeValues.empty = callee->getArgument(index);
          }
          barrierValues[calleeToken] = calleeValues;
          tokenOrigins[calleeToken] = origin;
          recordOriginalOperand(functionTokenArguments, callee->getOperation(),
                                operandNumber);
        } else {
          ttnvws::CreateTokenOp previous = tokenOrigins.lookup(calleeToken);
          if (!previous || !haveCompatibleTokenContracts(previous, origin))
            return call.emitOpError(
                "shared token-bearing callee has incompatible pipe contracts");
        }
        if (requirement & TokenNeedsFull)
          call->insertOperands(call.getNumOperands(), values.full);
        if (requirement & TokenNeedsEmpty)
          call->insertOperands(call.getNumOperands(), values.empty);
        recordOriginalOperand(callTokenOperands, call, operandNumber);
        if (failed(propagate(calleeToken)))
          return failure();
        continue;
      }
      return user->emitOpError("unexpected use of an NVWS token");
    }
    return success();
  };

  for (ttnvws::CreateTokenOp create : createTokens)
    if (failed(propagate(create.getResult())))
      return failure();

  SmallVector<Operation *> deprecatedOps;
  for (Operation *user : allLifecycleUsers) {
    Value token;
    if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user))
      token = op.getToken();
    else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user))
      token = op.getToken();
    else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user))
      token = op.getToken();
    else
      token = cast<ttnvws::ConsumerReleaseOp>(user).getToken();
    TokenBarrierValues values = barrierValues.lookup(token);
    ttnvws::CreateTokenOp origin = tokenOrigins.lookup(token);
    if (!origin)
      return user->emitOpError("has no propagated create-token origin");

    OpBuilder builder(user);
    Type slotType = ttg::MemDescType::get(
        {1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);
    auto fullSlot = [&](Value idx) -> Value {
      assert(values.full && "full-barrier user requires a full array");
      return ttg::MemDescIndexOp::create(builder, user->getLoc(), slotType,
                                         values.full, idx);
    };
    auto emptySlot = [&](Value idx) -> Value {
      assert(values.empty && "empty-barrier user requires an empty array");
      return ttg::MemDescIndexOp::create(builder, user->getLoc(), slotType,
                                         values.empty, idx);
    };

    if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user)) {
      Value barrier = emptySlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      processProducerAcquireOp(builder, op, barrier);
    } else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user)) {
      Value barrier = fullSlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      if (op.getCommitKind() ==
          ttnvws::ProducerCommitKind::TmaCopyBarrierArrive) {
        if (failed(processProducerCommitTmaCopyOp(builder, op, barrier)))
          return failure();
      } else {
        processProducerCommitOp(builder, op, barrier, origin.getLoadType(),
                                tokenFullCounts.lookup(origin.getOperation()));
      }
    } else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user)) {
      Value barrier = fullSlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      processConsumerWaitOp(builder, op, barrier);
    } else {
      auto releaseOp = cast<ttnvws::ConsumerReleaseOp>(user);
      Value barrier = emptySlot(releaseOp.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      unsigned releaseCount = THREADS_PER_TASK;
      if (auto attr =
              releaseOp->getAttrOfType<IntegerAttr>("release_count"))
        releaseCount = static_cast<unsigned>(attr.getInt());
      processConsumerReleaseOp(builder, releaseOp, barrier, numCTAs,
                               releaseCount);
    }
    deprecatedOps.push_back(user);
  }

  for (Operation *op : deprecatedOps)
    op->erase();

  auto eraseOriginalOperands = [](auto op, SmallVector<unsigned> indices) {
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices)
      op->eraseOperand(index);
  };
  for (auto &[op, indices] : callTokenOperands)
    eraseOriginalOperands(cast<tt::CallOp>(op), indices);
  for (auto &[op, indices] : functionTokenArguments) {
    auto function = cast<tt::FuncOp>(op);
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices) {
      if (!function.getArgument(index).use_empty())
        return function.emitOpError("lowered token argument still has uses");
      if (failed(function.eraseArgument(index)))
        return function.emitOpError("cannot erase lowered token argument");
    }
  }
  for (auto &[op, indices] : wsTokenOperands) {
    auto wsOp = cast<ttg::WarpSpecializeOp>(op);
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices) {
      for (Region *region : wsOp.getPartitionRegions()) {
        if (!region->getArgument(index).use_empty())
          return wsOp.emitOpError("lowered token capture still has uses");
        region->eraseArgument(index);
      }
      wsOp->eraseOperand(index);
    }
  }
  for (ttnvws::CreateTokenOp create : createTokens) {
    if (!create.getResult().use_empty())
      return create.emitOpError("lowered token still has uses");
    create.erase();
  }

  coalesceMBarrierInitBarriers(parentOp);
  return success();
}
#endif

#ifdef __TLE__
LogicalResult lowerTokenOperations(Operation *parentOp, int numCTAs,
                                   int numConsumerGroups) {
  return lowerTokenOperationsThroughCalls(parentOp, numCTAs);
}
#else
void lowerTokenOperations(Operation *parentOp, int numCTAs,
                          int numConsumerGroups) {
  SmallVector<Operation *> deprecatedOps;
  SmallVector<Operation *> deprecatedTokenOps;
  DenseSet<Operation *> warpSpecOps;
  DenseMap<Operation *, Value> tokenToFull;
  DenseMap<Operation *, Value> tokenToEmpty;
  DenseMap<Operation *, bool> tokenNeedsFull;
  DenseMap<Operation *, bool> tokenNeedsEmpty;
#ifdef __TLE__
  bool loweringFailed = false;
#endif
  parentOp->walk([&](ttnvws::CreateTokenOp createTokenOp) {
#ifdef __TLE__
    if (loweringFailed)
      return;
#endif
    ttnvws::TokenLoadType loadType = createTokenOp.getLoadType();
    MLIRContext *context = createTokenOp.getContext();
    OpBuilder builder(createTokenOp);
    Location loc = createTokenOp.getLoc();

    bool needsFullBarrier = false;
    bool needsEmptyBarrier = false;
    auto recordTokenUserBarriers = [&](Operation *user) {
      if (isa<ttnvws::ProducerCommitOp, ttnvws::ConsumerWaitOp>(user))
        needsFullBarrier = true;
      if (isa<ttnvws::ProducerAcquireOp, ttnvws::ConsumerReleaseOp>(user))
        needsEmptyBarrier = true;
    };
    for (OpOperand &use : createTokenOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      recordTokenUserBarriers(user);
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned opndNum = use.getOperandNumber();
        for (Region *region : wsOp.getPartitionRegions()) {
          BlockArgument tokenArg = region->getArgument(opndNum);
          for (Operation *tokenUser : tokenArg.getUsers())
            recordTokenUserBarriers(tokenUser);
        }
      }
    }
    tokenNeedsFull[createTokenOp.getOperation()] = needsFullBarrier;
    tokenNeedsEmpty[createTokenOp.getOperation()] = needsEmptyBarrier;

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto barrierCTALayout = ttg::CTAEncodingAttr::getDefault(context, 1);
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, {0}, barrierCTALayout);
    Type barrierMemDescType = ttg::MemDescType::get(
        {createTokenOp.getNumBuffers(), 1}, builder.getI64Type(),
        barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);
    Type singleBarrierMemDescType =
        ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    // These are created prior to warp_specialize.
    Value bufferFullArray;
    if (needsFullBarrier) {
      bufferFullArray = mlir::triton::gpu::LocalAllocOp::create(
          builder, loc, barrierMemDescType, Value());
      tokenToFull[createTokenOp.getOperation()] = bufferFullArray;
    }
    Value bufferEmptyArray;
    if (needsEmptyBarrier) {
      bufferEmptyArray = mlir::triton::gpu::LocalAllocOp::create(
          builder, loc, barrierMemDescType, Value());
      tokenToEmpty[createTokenOp.getOperation()] = bufferEmptyArray;
    }

    unsigned bufferFullCount =
        loadType == ttnvws::TokenLoadType::TMALoadOp ? 1 : THREADS_PER_TASK;
    if (loadType != ttnvws::TokenLoadType::TMALoadOp) {
      if (auto fullCount = getTokenCountOverride(createTokenOp, "full_count"))
        bufferFullCount = *fullCount;
    }
#ifdef __TLE__
    if (auto offset = getTokenCountOverride(createTokenOp,
                                            kTleInferFullCountOffsetAttr)) {
      std::optional<unsigned> inferredArriveCount;
      auto recordTokenUser = [&](Operation *user) -> LogicalResult {
        if (auto commit = dyn_cast<ttnvws::ProducerCommitOp>(user))
          return recordInferredParticipantCount(commit, inferredArriveCount);
        return success();
      };
      for (OpOperand &use : createTokenOp.getResult().getUses()) {
        Operation *user = use.getOwner();
        if (failed(recordTokenUser(user))) {
          loweringFailed = true;
          return;
        }
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
          unsigned opndNum = use.getOperandNumber();
          for (Region *region : wsOp.getPartitionRegions()) {
            BlockArgument tokenArg = region->getArgument(opndNum);
            for (Operation *tokenUser : tokenArg.getUsers()) {
              if (failed(recordTokenUser(tokenUser))) {
                loweringFailed = true;
                return;
              }
            }
          }
        }
      }
      if (!inferredArriveCount) {
        createTokenOp.emitOpError("requires an inferred participant producer "
                                  "commit for token full_count inference");
        loweringFailed = true;
        return;
      }
      bufferFullCount = *inferredArriveCount + *offset;
      createTokenOp->removeAttr(kTleInferFullCountOffsetAttr);
    }
#endif
    unsigned bufferEmptyCount = THREADS_PER_TASK;
    if (auto emptyCount = getTokenCountOverride(createTokenOp, "empty_count"))
      bufferEmptyCount = *emptyCount;
    for (unsigned i = 0; i < createTokenOp.getNumBuffers(); i++) {
      Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
      // EmptyView is used for ConsumerRelease and ProducerAcquire.
      // FullView is for ConsumerWait and ProducerCommit.
      if (needsFullBarrier) {
        Value barrierFullView = ttg::MemDescIndexOp::create(
            builder, loc, singleBarrierMemDescType, bufferFullArray, idx);
        ttng::InitBarrierOp::create(builder, loc, barrierFullView,
                                    bufferFullCount);
      }

      if (needsEmptyBarrier) {
        Value barrierEmptyView = ttg::MemDescIndexOp::create(
            builder, loc, singleBarrierMemDescType, bufferEmptyArray, idx);
        ttng::InitBarrierOp::create(builder, loc, barrierEmptyView,
                                    bufferEmptyCount);
      }
    }

    assert(numCTAs == 1 && "remote CTA is not supported yet");
    if (needsFullBarrier || needsEmptyBarrier)
      mlir::gpu::BarrierOp::create(builder, loc);

    // Helper function for extracting one index from bufferFullArray.
    auto extractBufferFull = [&](Location loc, Value idx) -> Value {
      assert(bufferFullArray && "token user requires a full barrier");
      return ttg::MemDescIndexOp::create(builder, loc, singleBarrierMemDescType,
                                         bufferFullArray, idx);
    };

    // Helper function for extracting one index from bufferEmptyArray.
    auto extractBufferEmpty = [&](Location loc, Value idx) -> Value {
      assert(bufferEmptyArray && "token user requires an empty barrier");
      return ttg::MemDescIndexOp::create(builder, loc, singleBarrierMemDescType,
                                         bufferEmptyArray, idx);
    };
    auto handleOneUser = [&](Operation *user) -> bool {
      // Here builder is at the user, make sure usage of values outside of
      // warp_specialize is via capture if user is in a partition region.
      // We need bufferFullArray and bufferEmptyArray.
      if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        auto pOp = user->getParentOp();
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        processProducerAcquireOp(builder, op, bufferEmpty);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        if (op.getCommitKind() ==
            ttnvws::ProducerCommitKind::TmaCopyBarrierArrive) {
          if (failed(processProducerCommitTmaCopyOp(builder, op, bufferFull)))
            llvm_unreachable("invalid tma_copy_barrier_arrive producer commit");
        } else {
          processProducerCommitOp(builder, op, bufferFull, loadType,
                                  bufferFullCount);
        }
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        processConsumerWaitOp(builder, op, bufferFull);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ConsumerReleaseOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        unsigned releaseCount = THREADS_PER_TASK;
        if (auto attr = op->getAttrOfType<IntegerAttr>("release_count"))
          releaseCount = static_cast<unsigned>(attr.getInt());
        processConsumerReleaseOp(builder, op, bufferEmpty, numCTAs,
                                 releaseCount);
        deprecatedOps.push_back(user);
        return true;
      }
      return false;
    };

    // Process token users: ProducerAcquireOp, ProducerCommitOp, ConsumerWaitOp,
    // and ConsumerReleaseOp.
    for (OpOperand &use : createTokenOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      bool handled = handleOneUser(user);
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned opndNum = use.getOperandNumber();
        // Handle the regions. Trace uses of the argument corresponding to the
        // captured value.
        for (Region *region : wsOp.getPartitionRegions()) {
          LDBG("-- region " << region->getNumArguments());
          auto tArg = region->getArgument(opndNum);
          for (Operation *tUser : tArg.getUsers()) {
            builder.setInsertionPoint(tUser);
            // Use of TokenOp via capture of warp_specialize.
            handleOneUser(tUser);
          }
        }
        warpSpecOps.insert(user);
      } else if (!handled) {
        llvm_unreachable("Unexpected user of token");
      }
    }

    deprecatedTokenOps.push_back(createTokenOp);
  });
#ifdef __TLE__
  if (loweringFailed)
    return failure();
#endif
  for (auto op : deprecatedOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    op->erase();
  }
  unsigned tokenRemoval = 0;
  // Map from tokenOp to bufferFullArray, bufferEmptyArray.
  // If a tokenOp is used by warp_specialize, remove it and add
  // buffer[Full|Empty]Array.

  for (auto op : deprecatedTokenOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    ++tokenRemoval;
    if (auto tokenOp = dyn_cast<ttnvws::CreateTokenOp>(op)) {
      bool needsFullBarrier = tokenNeedsFull.lookup(op);
      bool needsEmptyBarrier = tokenNeedsEmpty.lookup(op);
      // Check to see if it is used by warpSpec. If yes, eraseOperand and
      // eraseArgument.
      for (OpOperand &use : llvm::make_early_inc_range(tokenOp->getUses())) {
        Operation *user = use.getOwner();
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
          unsigned opndNum = use.getOperandNumber();
          LDBG("wsOp user numOperands: " << wsOp->getNumOperands() << " idx "
                                         << opndNum);

          LLVM_DEBUG({
            LDBG("prior to erasing " << tokenRemoval);
            parentOp->dump();
          });
          wsOp->eraseOperand(opndNum);
          Value full;
          if (needsFullBarrier) {
            full = tokenToFull[op];
            wsOp->insertOperands(wsOp.getNumOperands(), full);
          }
          Value empty;
          if (needsEmptyBarrier) {
            empty = tokenToEmpty[op];
            wsOp->insertOperands(wsOp.getNumOperands(), empty);
          }
          // Handle the regions.
          for (Region *region : wsOp.getPartitionRegions()) {
            LDBG("-- region " << region->getNumArguments());
            auto tArg = region->getArgument(opndNum);
            for (Operation *tUser : tArg.getUsers()) {
              LLVM_DEBUG({
                LDBG("user for arg");
                tUser->dump();
              });
            }
            region->eraseArgument(opndNum);
            if (needsFullBarrier) {
              BlockArgument arg =
                  region->addArgument(full.getType(), full.getLoc());
              replaceAllUsesInRegionWith(full, arg, *region);
            }
            if (needsEmptyBarrier) {
              BlockArgument arg =
                  region->addArgument(empty.getType(), empty.getLoc());
              replaceAllUsesInRegionWith(empty, arg, *region);
            }
          }
        }
      }
    }
    op->erase();
  }
  coalesceMBarrierInitBarriers(parentOp);

  assert(numCTAs == 1 && "remote CTA is not supported yet");
  LLVM_DEBUG({
    LDBG("after lowering");
    parentOp->dump();
  });
}
#endif

#ifdef __TLE__
LogicalResult doTokenLowering(triton::FuncOp &funcOp,
                              unsigned numConsumerGroups)
#else
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups)
#endif
{
  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);

  // lowerGetAsyncTaskIdOp(mod, numConsumerGroups);
#ifdef __TLE__
  return lowerTokenOperations(mod, numCTAs, numConsumerGroups);
#else
  lowerTokenOperations(mod, numCTAs, numConsumerGroups);
#endif
}

} // namespace mlir
