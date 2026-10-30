#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "MUSATLE/Transforms/TransformAttrs.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;
namespace musa_tle = triton::musa_tle;

static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";

struct PartitionSync {
  LLVM::CallIntrinsicOp op;
  int32_t numWarps;
  Region *partition;
  int32_t sharedGroup = -1;
};

static FailureOr<Value> getForwardedBlockArgument(Block *predecessor,
                                                  Block *successor,
                                                  unsigned argument) {
  auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
  if (!branch)
    return failure();
  auto successorIt = llvm::find(branch->getSuccessors(), successor);
  if (successorIt == branch->getSuccessors().end())
    return failure();
  unsigned successorIndex =
      std::distance(branch->getSuccessors().begin(), successorIt);
  SuccessorOperands operands = branch.getSuccessorOperands(successorIndex);
  if (argument < operands.getProducedOperandCount())
    return failure();
  ValueRange forwarded = operands.getForwardedOperands();
  unsigned forwardedIndex = argument - operands.getProducedOperandCount();
  if (forwardedIndex >= forwarded.size())
    return failure();
  return forwarded[forwardedIndex];
}

static bool isConstant(Value value, int64_t expected) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant)))
    return false;
  auto integer = dyn_cast<IntegerAttr>(constant);
  return integer && integer.getInt() == expected;
}

static FailureOr<std::pair<Value, int64_t>>
getPositiveConstantIncrement(Value value, Value induction) {
  auto match = [&](Value lhs,
                   Value rhs) -> FailureOr<std::pair<Value, int64_t>> {
    Value increment;
    if (lhs == induction)
      increment = rhs;
    else if (rhs == induction)
      increment = lhs;
    else
      return failure();

    APInt constant;
    if (!matchPattern(increment, m_ConstantInt(&constant)) ||
        constant.getSExtValue() <= 0)
      return failure();
    return std::make_pair(increment, constant.getSExtValue());
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return match(add.getLhs(), add.getRhs());
  if (auto add = value.getDefiningOp<LLVM::AddOp>())
    return match(add.getLhs(), add.getRhs());
  return failure();
}

// scf.for has become CFG by late warp-specialize lowering.  Recover the
// canonical induction variable, lower bound, and positive constant step so
// every consumer in a shared epoch group waits on the same alternating
// hardware phase.  In particular, a grid-stride persistent loop uses
// ((tile - pid) / NUM_SMS) & 1 rather than tile & 1.
static FailureOr<Value> resolvePartitionSyncPhase(LLVM::CallIntrinsicOp sync,
                                                  Region &partition,
                                                  DominanceInfo &dominance,
                                                  IRRewriter &rewriter) {
  Block *loopHeader = nullptr;
  Value loopInitial;
  Value loopStep;
  int64_t loopStepValue = 0;
  for (Block &candidate : partition.getBlocks()) {
    if (candidate.getNumArguments() == 0 ||
        !dominance.dominates(&candidate, sync->getBlock()))
      continue;
    Value induction = candidate.getArgument(0);
    auto inductionType = dyn_cast<IntegerType>(induction.getType());
    if (!inductionType || inductionType.getWidth() != 32)
      continue;

    SmallVector<Block *> entries;
    SmallVector<Block *> backedges;
    for (Block *predecessor : candidate.getPredecessors()) {
      (dominance.dominates(&candidate, predecessor) ? backedges : entries)
          .push_back(predecessor);
    }
    if (entries.size() != 1 || backedges.size() != 1)
      continue;
    Block *backedge = backedges.front();
    // The epoch marker is inserted at the outer-loop latch.  It must
    // therefore dominate the selected loop's backedge.  A completed nested
    // K loop can dominate the marker, but the marker cannot dominate that
    // inner backedge, which keeps its induction variable out of the phase.
    if (!dominance.dominates(sync->getBlock(), backedge))
      continue;
    FailureOr<Value> initial =
        getForwardedBlockArgument(entries.front(), &candidate, 0);
    FailureOr<Value> next = getForwardedBlockArgument(backedge, &candidate, 0);
    if (failed(initial) || failed(next))
      continue;
    FailureOr<std::pair<Value, int64_t>> increment =
        getPositiveConstantIncrement(*next, induction);
    if (failed(increment))
      continue;

    if (loopHeader)
      return failure();
    loopHeader = &candidate;
    loopInitial = *initial;
    loopStep = (*increment).first;
    loopStepValue = (*increment).second;
  }

  if (!loopHeader)
    return failure();
  rewriter.setInsertionPoint(sync);
  Value iteration = loopHeader->getArgument(0);
  if (!isConstant(loopInitial, 0))
    iteration =
        arith::SubIOp::create(rewriter, sync.getLoc(), iteration, loopInitial)
            .getResult();
  if (loopStepValue != 1)
    iteration =
        arith::DivUIOp::create(rewriter, sync.getLoc(), iteration, loopStep)
            .getResult();

  Value one = arith::ConstantIntOp::create(rewriter, sync.getLoc(), 1, 32);
  return arith::AndIOp::create(rewriter, sync.getLoc(), iteration, one)
      .getResult();
}

static LogicalResult lowerWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                            ttg::WarpSpecializeOp ws,
                                            IRRewriter &rewriter) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!defaultWarpsAttr || defaultWarpsAttr.getInt() <= 0 ||
      defaultWarpsAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "MUSA TLE default partition requires a positive int32 "
        "ttg.num-warps");
  int32_t defaultWarps = static_cast<int32_t>(defaultWarpsAttr.getInt());
  bool needsLmaCompletion = false;
  if (auto target = module->getAttrOfType<StringAttr>(ttg::AttrTargetName)) {
    StringRef arch = target.getValue();
    int capability = 0;
    if (arch.consume_front("musa:"))
      needsLmaCompletion =
          arch.starts_with("ph1") ||
          (!arch.getAsInteger(10, capability) && capability >= 31);
  }

  SmallVector<Region *> workerRegions(ws.getPartitionRegions().begin(),
                                      ws.getPartitionRegions().end());
  ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
  if (workerRegions.empty() || workerRegions.size() != workerWarps.size())
    return ws.emitOpError(
        "MUSA TLE partition synchronization requires matching worker "
        "regions and warp counts");

  // The regular layout assigns the final worker to the producer.
  // Every non-explicit local barrier that survives pipe dependency analysis
  // is a real partition rendezvous. Explicit frontend CTA barriers remain
  // untouched and are lowered after the static partitions are flattened.
  Region &producerRegion = *workerRegions.back();
  int32_t producerWarps = workerWarps.back();
  SmallVector<PartitionSync> syncs;
  SmallVector<LLVM::CallIntrinsicOp> tmeReuseSyncs;
  producerRegion.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() != kLocalSyncIntrinsic ||
        call->hasAttr(musa_tle::kMUSATLEExplicitCtaSyncAttr))
      return;
    if (needsLmaCompletion &&
        call->hasAttr(musa_tle::kMUSATLEPipeTMEReuseSyncAttr)) {
      tmeReuseSyncs.push_back(call);
      return;
    }
    syncs.push_back({call, producerWarps, &producerRegion, -1});
  });

  bool invalidSharedGroup = false;
  auto collectConsumerSyncs = [&](Region &region, int32_t warps) {
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
      if (!call)
        return WalkResult::advance();
      if (call.getIntrin() != kLocalSyncIntrinsic ||
          call->hasAttr(musa_tle::kMUSATLEExplicitCtaSyncAttr))
        return WalkResult::advance();
      if (auto group = call->getAttrOfType<IntegerAttr>(
              musa_tle::kMUSATLEConsumerEpochSyncAttr)) {
        if (group.getInt() < 0 ||
            group.getInt() > std::numeric_limits<int32_t>::max()) {
          invalidSharedGroup = true;
          return WalkResult::interrupt();
        }
        syncs.push_back(
            {call, warps, &region, static_cast<int32_t>(group.getInt())});
        return WalkResult::advance();
      }
      syncs.push_back({call, warps, &region, -1});
      return WalkResult::advance();
    });
  };
  collectConsumerSyncs(ws.getDefaultRegion(), defaultWarps);
  for (unsigned index = 0; index + 1 < workerRegions.size(); ++index)
    collectConsumerSyncs(*workerRegions[index], workerWarps[index]);

  if (invalidSharedGroup)
    return ws.emitOpError(
        "MUSA TLE consumer epoch synchronization group must be a "
        "non-negative int32");

  // The bundled LLVM does not register this newer MCC intrinsic. Declare it
  // as an external call, as for memcpy.g2s; MCC resolves the intrinsic later.
  LLVM::LLVMFuncOp lmaWait;
  if (needsLmaCompletion && (!syncs.empty() || !tmeReuseSyncs.empty())) {
    StringRef name = "llvm.musa.lma.wait";
    lmaWait = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!lmaWait) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(module.getContext()), {});
      lmaWait = LLVM::LLVMFuncOp::create(rewriter, ws.getLoc(), name, type);
    }
  }

  // Membar inserts this sync for the loop-carried WAW on a pipe-managed TME
  // payload.  PH1 still needs the local-memory completion edge, but turning
  // it into a partition rendezvous allocates an unnecessary hardware barrier.
  // Keep the completion semantics with the target's LMA wait instead.
  for (LLVM::CallIntrinsicOp sync : tmeReuseSyncs) {
    rewriter.setInsertionPoint(sync);
    LLVM::CallOp::create(rewriter, sync.getLoc(), lmaWait, ValueRange{});
    rewriter.eraseOp(sync);
  }

  if (syncs.empty())
    return success();

  llvm::DenseSet<Region *> privatePartitions;
  llvm::DenseMap<int32_t, int32_t> sharedGroupWarps;
  llvm::DenseMap<int32_t, llvm::DenseSet<Region *>> sharedGroupPartitions;
  SmallVector<int32_t> sharedGroupOrder;
  for (PartitionSync &sync : syncs) {
    if (sync.sharedGroup < 0) {
      privatePartitions.insert(sync.partition);
      continue;
    }
    auto &partitions = sharedGroupPartitions[sync.sharedGroup];
    if (!partitions.insert(sync.partition).second)
      return sync.op.emitOpError(
          "MUSA TLE consumer epoch sync group contains duplicate markers "
          "for one partition");
    auto it = sharedGroupWarps.find(sync.sharedGroup);
    if (it == sharedGroupWarps.end()) {
      sharedGroupWarps[sync.sharedGroup] = sync.numWarps;
      sharedGroupOrder.push_back(sync.sharedGroup);
      continue;
    }
    if (it->second > std::numeric_limits<int32_t>::max() - sync.numWarps)
      return sync.op.emitOpError(
          "MUSA TLE shared consumer epoch warp count overflow");
    it->second += sync.numWarps;
  }

  if (!sharedGroupOrder.empty()) {
    ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
    llvm::DenseSet<Region *> expectedConsumerPartitions;
    expectedConsumerPartitions.insert(&ws.getDefaultRegion());
    int64_t expectedConsumerWarps = defaultWarps;
    // The regular layout places the producer in the final worker partition.
    // Every preceding partition participates in the shared consumer epoch
    // rendezvous.
    for (unsigned index = 0; index + 1 < workerWarps.size(); ++index) {
      expectedConsumerWarps += workerWarps[index];
      expectedConsumerPartitions.insert(ws.getPartitionRegions()[index]);
    }
    if (expectedConsumerWarps <= 0 ||
        expectedConsumerWarps > std::numeric_limits<int32_t>::max())
      return syncs.front().op.emitOpError(
          "MUSA TLE total consumer warp count overflow");
    for (int32_t group : sharedGroupOrder) {
      const llvm::DenseSet<Region *> &actual =
          sharedGroupPartitions.lookup(group);
      bool samePartitions =
          actual.size() == expectedConsumerPartitions.size() &&
          llvm::all_of(expectedConsumerPartitions,
                       [&](Region *region) { return actual.contains(region); });
      if (!samePartitions ||
          sharedGroupWarps.lookup(group) != expectedConsumerWarps)
        return syncs.front().op.emitOpError(
            "MUSA TLE consumer epoch sync must cover every consumer "
            "partition exactly once");
    }
  }

  int64_t requiredBarrierIds =
      privatePartitions.size() + sharedGroupOrder.size();
  if (requiredBarrierIds <= 0 ||
      requiredBarrierIds > std::numeric_limits<int32_t>::max())
    return syncs.front().op.emitOpError(
        "MUSA TLE partition synchronization barrier count overflow");

  // Each thread executes exactly one static partition, which reuses one
  // hardware barrier. Track its phase explicitly across every rendezvous,
  // including loops and request boundaries. LLVM promotes this private slot
  // to SSA, matching MATE's explicit phase protocol on PH1.
  Value phaseSlot;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&func.getBody().front());
    Value one = arith::ConstantIntOp::create(rewriter, ws.getLoc(), 1, 32);
    Value zero = arith::ConstantIntOp::create(rewriter, ws.getLoc(), 0, 32);
    phaseSlot = LLVM::AllocaOp::create(
        rewriter, ws.getLoc(),
        LLVM::LLVMPointerType::get(module.getContext(), 5),
        rewriter.getI32Type(), one, 4);
    LLVM::StoreOp::create(rewriter, ws.getLoc(), zero, phaseSlot);
  }

  auto reserved = ttmg::reserveBarrierIdRange(
      syncs.front().op, static_cast<int32_t>(requiredBarrierIds));
  if (failed(reserved))
    return syncs.front().op.emitOpError(
        "MUSA TLE partition synchronization exhausted hardware barrier "
        "ids");

  LLVM::CallIntrinsicOp initializationRendezvous;
  for (Operation *op = ws->getPrevNode(); op; op = op->getPrevNode()) {
    auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
    if (call && call.getIntrin() == kLocalSyncIntrinsic) {
      initializationRendezvous = call;
      break;
    }
  }

  Location loc = initializationRendezvous ? initializationRendezvous.getLoc()
                                          : ws.getLoc();
  if (initializationRendezvous)
    rewriter.setInsertionPoint(initializationRendezvous);
  else
    rewriter.setInsertionPoint(ws);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  DominanceInfo dominance(func);
  llvm::DenseMap<Region *, Value> privateBarrierIds;
  llvm::DenseMap<int32_t, Value> sharedBarrierIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (PartitionSync &sync : syncs) {
    if (sync.sharedGroup >= 0 || privateBarrierIds.count(sync.partition))
      continue;
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, sync.numWarps, 32);
    initializationArgs.push_back({id, count});
    privateBarrierIds[sync.partition] = id;
  }
  for (int32_t group : sharedGroupOrder) {
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count = arith::ConstantIntOp::create(
        rewriter, loc, sharedGroupWarps.lookup(group), 32);
    initializationArgs.push_back({id, count});
    sharedBarrierIds[group] = id;
  }

  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value issueInit = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::eq, tid, phase);
  auto initIf = scf::IfOp::create(rewriter, loc, issueInit, false);
  rewriter.setInsertionPointToStart(&initIf.getThenRegion().front());
  for (auto [id, count] : initializationArgs)
    LLVM::CallIntrinsicOp::create(
        rewriter, loc, rewriter.getStringAttr("llvm.musa.async.init.arrival"),
        ValueRange{id, count, phase});

  rewriter.setInsertionPointAfter(initIf);
  if (!initializationRendezvous)
    LLVM::CallIntrinsicOp::create(rewriter, loc,
                                  rewriter.getStringAttr(kLocalSyncIntrinsic),
                                  ValueRange{});

  for (PartitionSync &sync : syncs) {
    rewriter.setInsertionPoint(sync.op);
    Value id = sync.sharedGroup >= 0 ? sharedBarrierIds.lookup(sync.sharedGroup)
                                     : privateBarrierIds.lookup(sync.partition);
    // syncthreads.lm also completes preceding local-memory accesses. A named
    // arrival/wait alone loses that ordering on PH1, including when vector
    // loads are still outstanding as the shared buffer is reused.
    if (needsLmaCompletion)
      LLVM::CallOp::create(rewriter, sync.op.getLoc(), lmaWait, ValueRange{});
    Value arrived;
    if (sync.sharedGroup >= 0) {
      FailureOr<Value> resolved = resolvePartitionSyncPhase(
          sync.op, *sync.partition, dominance, rewriter);
      if (failed(resolved))
        return sync.op.emitOpError(
            "MUSA TLE consumer epoch sync requires exactly one canonical "
            "outer-loop induction variable");
      arrived = *resolved;
    } else {
      arrived = LLVM::LoadOp::create(rewriter, sync.op.getLoc(),
                                     rewriter.getI32Type(), phaseSlot)
                    .getResult();
    }
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.arrive.none.phaseid"),
        ValueRange{id});
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.wait"),
        ValueRange{id, arrived});
    if (sync.sharedGroup < 0) {
      Value one =
          arith::ConstantIntOp::create(rewriter, sync.op.getLoc(), 1, 32);
      Value nextPhase =
          arith::XOrIOp::create(rewriter, sync.op.getLoc(), arrived, one);
      LLVM::StoreOp::create(rewriter, sync.op.getLoc(), nextPhase, phaseSlot);
    }
    rewriter.eraseOp(sync.op);
  }

  SmallVector<LLVM::CallIntrinsicOp> oldBarRecords;
  func.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() == kBarRecordIntrinsic)
      oldBarRecords.push_back(call);
  });
  for (LLVM::CallIntrinsicOp record : oldBarRecords)
    rewriter.eraseOp(record);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Value barCount = arith::ConstantIntOp::create(
      rewriter, func.getLoc(), ttmg::getReservedBarrierCount(func), 32);
  LLVM::CallIntrinsicOp::create(rewriter, func.getLoc(),
                                rewriter.getStringAttr(kBarRecordIntrinsic),
                                ValueRange{barCount});
  return success();
}

static LogicalResult lowerStaticWarpSpecialize(LLVM::LLVMFuncOp func,
                                               ttg::WarpSpecializeOp ws,
                                               IRRewriter &rewriter) {
  auto workerRegions = ws.getPartitionRegions();
  ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
  if (ws.getNumResults() != 0 || workerRegions.empty() ||
      workerRegions.size() != workerWarps.size())
    return ws.emitOpError(
        "MUSA TLE late lowering requires a default partition, one or more "
        "worker partitions, and no results");

  Region &defaultRegion = ws.getDefaultRegion();
  if (defaultRegion.empty())
    return ws.emitOpError(
        "MUSA TLE static default partition must not be empty");
  for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
    if (workerRegion->empty())
      return ws.emitOpError() << "MUSA TLE static worker partition #"
                              << workerIndex << " must not be empty";
  }

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
    Block &workerEntry = workerRegion->front();
    if (workerEntry.getNumArguments() != captures.size())
      return ws.emitOpError() << "MUSA TLE worker partition #" << workerIndex
                              << " capture count changed during lowering";
    for (auto [argument, capture] :
         llvm::zip(workerEntry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError()
               << "MUSA TLE worker partition #" << workerIndex
               << " capture types were not converted consistently";
      argument.replaceAllUsesWith(capture);
    }
    workerEntry.eraseArguments([](BlockArgument) { return true; });
  }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  auto totalWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  auto workerStartIds = ws.getWarpGroupStartIds();
  if (!defaultWarpsAttr || !threadsPerWarpAttr || !totalWarpsAttr ||
      !workerStartIds || defaultWarpsAttr.getInt() <= 0 ||
      threadsPerWarpAttr.getInt() <= 0 || totalWarpsAttr.getInt() <= 0)
    return ws.emitOpError(
        "MUSA TLE late lowering requires positive ttg.num-warps, "
        "ttg.threads-per-warp, ttg.total-num-warps, and worker start ids");
  if (workerStartIds->size() != workerRegions.size())
    return ws.emitOpError(
        "MUSA TLE late lowering requires one start id per worker partition");

  int64_t nextWarp = defaultWarpsAttr.getInt();
  SmallVector<int32_t> threadEnds;
  threadEnds.reserve(workerRegions.size() + 1);
  auto appendThreadEnd = [&](int64_t warpEnd) -> LogicalResult {
    if (warpEnd <= 0 || warpEnd > std::numeric_limits<int32_t>::max() /
                                      threadsPerWarpAttr.getInt())
      return ws.emitOpError(
          "MUSA TLE static partition boundary exceeds int32 range");
    threadEnds.push_back(
        static_cast<int32_t>(warpEnd * threadsPerWarpAttr.getInt()));
    return success();
  };
  if (failed(appendThreadEnd(nextWarp)))
    return failure();
  for (auto [workerIndex, startAndCount] :
       llvm::enumerate(llvm::zip(*workerStartIds, workerWarps))) {
    auto [startId, numWarps] = startAndCount;
    if (startId != nextWarp || numWarps <= 0)
      return ws.emitOpError() << "MUSA TLE worker partition #" << workerIndex
                              << " has a non-contiguous static warp range";
    nextWarp += numWarps;
    if (failed(appendThreadEnd(nextWarp)))
      return failure();
  }
  if (nextWarp != totalWarpsAttr.getInt())
    return ws.emitOpError(
        "MUSA TLE worker warp ranges do not match ttg.total-num-warps");

  SmallVector<SmallVector<ttg::WarpReturnOp>> workerReturns;
  SmallVector<Block *> workerRoots;
  workerReturns.reserve(workerRegions.size());
  workerRoots.reserve(workerRegions.size());
  for (Region *workerRegion : workerRegions) {
    workerRoots.push_back(&workerRegion->front());
    auto &returns = workerReturns.emplace_back();
    workerRegion->walk([&](ttg::WarpReturnOp op) { returns.push_back(op); });
    if (returns.empty())
      return ws.emitOpError(
          "MUSA TLE static worker partition lost its terminator");
  }
  SmallVector<ttg::WarpYieldOp> defaultYields;
  defaultRegion.walk([&](ttg::WarpYieldOp op) { defaultYields.push_back(op); });
  if (defaultYields.empty())
    return ws.emitOpError(
        "MUSA TLE static default partition lost its terminator");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  Block *defaultRoot = &defaultRegion.front();

  if (workerRegions.size() == 1) {
    Region &workerRegion = *workerRegions.front();
    funcBlocks.splice(continuation->getIterator(), workerRegion.getBlocks());
    Block *producerJoin =
        rewriter.createBlock(&funcBody, continuation->getIterator());
    funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());

    Block *producerRoot = workerRoots.front();
    Block *consumerRoot = defaultRoot;
    SmallVector<Operation *> producerTerminators;
    SmallVector<Operation *> consumerTerminators;
    for (ttg::WarpReturnOp op : workerReturns.front())
      producerTerminators.push_back(op);
    for (ttg::WarpYieldOp op : defaultYields)
      consumerTerminators.push_back(op);

    for (Operation *op : producerTerminators) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op->getLoc(), producerJoin);
      rewriter.eraseOp(op);
    }
    for (Operation *op : consumerTerminators) {
      if (auto yield = dyn_cast<ttg::WarpYieldOp>(op);
          yield && yield.getNumOperands() != 0)
        return op->emitOpError(
            "MUSA TLE static default partition must not yield values");
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op->getLoc(), continuation);
      rewriter.eraseOp(op);
    }

    Location loc = ws.getLoc();
    rewriter.eraseOp(ws);
    rewriter.setInsertionPointToEnd(dispatch);
    Value tid = LLVM::CallIntrinsicOp::create(
                    rewriter, loc, rewriter.getI32Type(),
                    rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"),
                    ValueRange{})
                    .getResult(0);
    Value boundaryValue =
        arith::ConstantIntOp::create(rewriter, loc, threadEnds.front(), 32);
    Value isProducer = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::uge, tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isProducer, producerRoot,
                             ValueRange{}, producerJoin, ValueRange{});

    rewriter.setInsertionPointToEnd(producerJoin);
    Value isConsumer = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isConsumer, consumerRoot,
                             ValueRange{}, continuation, ValueRange{});
    return success();
  }

  for (Region *workerRegion : workerRegions)
    funcBlocks.splice(continuation->getIterator(), workerRegion->getBlocks());
  funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());

  for (auto &returns : workerReturns) {
    for (ttg::WarpReturnOp op : returns) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op.getLoc(), continuation);
      rewriter.eraseOp(op);
    }
  }
  for (ttg::WarpYieldOp op : defaultYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "MUSA TLE static default partition must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  SmallVector<Block *> rangeTargets;
  rangeTargets.reserve(workerRoots.size() + 1);
  rangeTargets.push_back(defaultRoot);
  rangeTargets.append(workerRoots.begin(), workerRoots.end());
  SmallVector<Block *> fallbackDispatches;
  fallbackDispatches.reserve(rangeTargets.size() - 1);
  for (size_t index = 1; index < rangeTargets.size(); ++index)
    fallbackDispatches.push_back(
        rewriter.createBlock(&funcBody, continuation->getIterator()));

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  for (size_t index = 0; index < rangeTargets.size(); ++index) {
    if (index != 0)
      rewriter.setInsertionPointToEnd(fallbackDispatches[index - 1]);
    Value end =
        arith::ConstantIntOp::create(rewriter, loc, threadEnds[index], 32);
    Value inRange = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::ult, tid, end);
    Block *fallback = index + 1 < rangeTargets.size()
                          ? fallbackDispatches[index]
                          : continuation;
    cf::CondBranchOp::create(rewriter, loc, inRange, rangeTargets[index],
                             ValueRange{}, fallback, ValueRange{});
  }
  return success();
}

class LowerWarpSpecializePass
    : public impl::TritonMUSAGPUTLELowerWarpSpecializeBase<
          LowerWarpSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto stripInternalSyncMarkers = [&]() {
      module.walk([&](LLVM::CallIntrinsicOp call) {
        call->removeAttr(musa_tle::kMUSATLEExplicitCtaSyncAttr);
        call->removeAttr(musa_tle::kMUSATLEPipeTMEReuseSyncAttr);
        call->removeAttr(musa_tle::kMUSATLELayoutConversionSyncAttr);
      });
    };
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(musa_tle::kMUSATLEStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty()) {
      stripInternalSyncMarkers();
      return;
    }
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "MUSA TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    auto func = marked.front()->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      marked.front().emitOpError(
          "MUSA TLE late lowering requires an LLVM function");
      return signalPassFailure();
    }

    // Unknown CTA-wide synchronization cannot be narrowed safely. Diagnose it
    // while the static partition and source location are still available.
    WalkResult checked = marked.front()->walk([&](LLVM::CallIntrinsicOp call) {
      if (call.getIntrin() != "llvm.musa.barrier0")
        return WalkResult::advance();
      call.emitOpError(
          "CTA barrier inside MUSA TLE static warp_specialize partition "
          "would wait for unrelated warps; use partition synchronization");
      return WalkResult::interrupt();
    });
    if (checked.wasInterrupted())
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    if (failed(lowerWarpGroupBarriers(func, marked.front(), rewriter)) ||
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter))) {
      signalPassFailure();
      return;
    }
    stripInternalSyncMarkers();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
