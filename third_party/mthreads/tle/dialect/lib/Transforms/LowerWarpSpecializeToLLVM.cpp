#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
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

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
static constexpr StringLiteral kExplicitCtaSyncAttr =
    "musa_tle.explicit_cta_sync";
static constexpr StringLiteral kLayoutConversionSyncAttr =
    "musa_tle.layout_conversion_sync";
static constexpr StringLiteral kConsumerEpochSyncAttr =
    "musa_tle.consumer_epoch_sync";
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";
static constexpr StringLiteral kDefaultProducerAttr =
    "musa_tle.default_producer";

struct PartitionSync {
  LLVM::CallIntrinsicOp op;
  int32_t numWarps;
  int32_t sharedGroup = -1;
};

static bool isSqmmaMmaIntrinsic(StringRef name) {
  return name.starts_with("llvm.musa.sqmma.") && name.ends_with(".mma");
}

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

static bool isUnitIncrement(Value value, Value induction) {
  auto matches = [&](Value lhs, Value rhs) {
    return (lhs == induction && isConstant(rhs, 1)) ||
           (rhs == induction && isConstant(lhs, 1));
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return matches(add.getLhs(), add.getRhs());
  if (auto add = value.getDefiningOp<LLVM::AddOp>())
    return matches(add.getLhs(), add.getRhs());
  return false;
}

// scf.for has already become CFG when static warp specialization is lowered.
// Recover the canonical 0-based unit-step induction variable so a partition
// barrier reused by the loop waits on alternating hardware phases.
static Value resolvePartitionSyncPhase(LLVM::CallIntrinsicOp sync,
                                       Region &partition,
                                       DominanceInfo &dominance,
                                       IRRewriter &rewriter,
                                       Value initialPhase) {
  Block *loopHeader = nullptr;
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
    FailureOr<Value> initial =
        getForwardedBlockArgument(entries.front(), &candidate, 0);
    FailureOr<Value> next =
        getForwardedBlockArgument(backedges.front(), &candidate, 0);
    if (failed(initial) || failed(next) || !isConstant(*initial, 0) ||
        !isUnitIncrement(*next, induction))
      continue;

    if (!loopHeader || dominance.dominates(loopHeader, &candidate))
      loopHeader = &candidate;
  }

  if (!loopHeader)
    return initialPhase;
  rewriter.setInsertionPoint(sync);
  Value one = arith::ConstantIntOp::create(rewriter, sync.getLoc(), 1, 32);
  return arith::AndIOp::create(rewriter, sync.getLoc(),
                               loopHeader->getArgument(0), one);
}

static LogicalResult lowerWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                            ttg::WarpSpecializeOp ws,
                                            IRRewriter &rewriter) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!consumerWarpsAttr || consumerWarpsAttr.getInt() <= 0 ||
      consumerWarpsAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "mthreads TLE default partition requires a positive int32 "
        "ttg.num-warps");
  int32_t consumerWarps = static_cast<int32_t>(consumerWarpsAttr.getInt());

  SmallVector<Region *> partitionRegions(ws.getPartitionRegions().begin(),
                                         ws.getPartitionRegions().end());
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (partitionRegions.empty() ||
      partitionRegions.size() != partitionNumWarps.size())
    return ws.emitOpError(
        "mthreads TLE partition synchronization requires matching worker "
        "regions and warp counts");
  bool defaultProducer = ws->hasAttr(kDefaultProducerAttr);
  Region &producerRegion = defaultProducer ? ws.getDefaultRegion()
                                           : *partitionRegions.back();
  SmallVector<LLVM::CallIntrinsicOp> redundantSyncs;
  SmallVector<PartitionSync> syncs;

  int32_t producerWarps =
      defaultProducer ? consumerWarps : partitionNumWarps.back();

  producerRegion.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() != kLocalSyncIntrinsic ||
        call->hasAttr(kExplicitCtaSyncAttr))
      return;
    if (call->hasAttr(kLayoutConversionSyncAttr))
      syncs.push_back({call, producerWarps});
    else
      redundantSyncs.push_back(call);
  });

  auto collectConsumerSyncs = [&](Region &region, int32_t numWarps) {
    bool seenSqmma = false;
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
      if (!call)
        return WalkResult::advance();
      if (isSqmmaMmaIntrinsic(call.getIntrin())) {
        seenSqmma = true;
        return WalkResult::advance();
      }
      if (call.getIntrin() != kLocalSyncIntrinsic)
        return WalkResult::advance();
      if (call->hasAttr(kExplicitCtaSyncAttr))
        return WalkResult::advance();
      if (auto group = call->getAttrOfType<IntegerAttr>(
              kConsumerEpochSyncAttr)) {
        if (group.getInt() < 0 ||
            group.getInt() > std::numeric_limits<int32_t>::max())
          return WalkResult::interrupt();
        syncs.push_back(
            {call, numWarps, static_cast<int32_t>(group.getInt())});
        return WalkResult::advance();
      }
      if (seenSqmma || call->hasAttr(kLayoutConversionSyncAttr))
        syncs.push_back({call, numWarps});
      else
        redundantSyncs.push_back(call);
      return WalkResult::advance();
    });
  };
  if (defaultProducer) {
    for (auto [region, numWarps] :
         llvm::zip(partitionRegions, partitionNumWarps))
      collectConsumerSyncs(*region, numWarps);
  } else {
    collectConsumerSyncs(ws.getDefaultRegion(), consumerWarps);
    for (unsigned index = 0; index + 1 < partitionRegions.size(); ++index)
      collectConsumerSyncs(*partitionRegions[index], partitionNumWarps[index]);
  }

  for (LLVM::CallIntrinsicOp redundant : redundantSyncs)
    rewriter.eraseOp(redundant);
  if (syncs.empty())
    return success();

  llvm::DenseMap<int32_t, int32_t> sharedGroupWarps;
  SmallVector<int32_t> sharedGroupOrder;
  int64_t requiredBarrierIds = 0;
  for (PartitionSync &sync : syncs) {
    if (sync.sharedGroup < 0) {
      ++requiredBarrierIds;
      continue;
    }
    auto it = sharedGroupWarps.find(sync.sharedGroup);
    if (it == sharedGroupWarps.end()) {
      sharedGroupWarps[sync.sharedGroup] = sync.numWarps;
      sharedGroupOrder.push_back(sync.sharedGroup);
      ++requiredBarrierIds;
      continue;
    }
    if (it->second > std::numeric_limits<int32_t>::max() - sync.numWarps)
      return sync.op.emitOpError(
          "mthreads TLE shared consumer epoch warp count overflow");
    it->second += sync.numWarps;
  }
  int64_t expectedConsumerWarps = defaultProducer ? 0 : consumerWarps;
  if (defaultProducer) {
    for (int32_t warps : partitionNumWarps)
      expectedConsumerWarps += warps;
  } else {
    for (unsigned index = 0; index + 1 < partitionNumWarps.size(); ++index)
      expectedConsumerWarps += partitionNumWarps[index];
  }
  if (expectedConsumerWarps <= 0 ||
      expectedConsumerWarps > std::numeric_limits<int32_t>::max())
    return syncs.front().op.emitOpError(
        "mthreads TLE total consumer warp count overflow");
  for (int32_t group : sharedGroupOrder)
    if (sharedGroupWarps.lookup(group) != expectedConsumerWarps)
      return syncs.front().op.emitOpError(
          "mthreads TLE consumer epoch sync must cover every consumer warp");
  if (requiredBarrierIds <= 0 ||
      requiredBarrierIds > std::numeric_limits<int32_t>::max())
    return syncs.front().op.emitOpError(
        "mthreads TLE partition synchronization barrier count overflow");

  auto reserved = ttmg::reserveBarrierIdRange(
      syncs.front().op, static_cast<int32_t>(requiredBarrierIds));
  if (failed(reserved))
    return syncs.front().op.emitOpError(
        "mthreads TLE partition synchronization exhausted hardware barrier "
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
  llvm::DenseMap<Operation *, Value> barrierIds;
  llvm::DenseMap<int32_t, Value> sharedGroupIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (PartitionSync &sync : syncs) {
    if (sync.sharedGroup >= 0)
      continue;
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, sync.numWarps, 32);
    initializationArgs.push_back({id, count});
    barrierIds[sync.op.getOperation()] = id;
  }
  for (int32_t group : sharedGroupOrder) {
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count = arith::ConstantIntOp::create(
        rewriter, loc, sharedGroupWarps.lookup(group), 32);
    initializationArgs.push_back({id, count});
    sharedGroupIds[group] = id;
  }
  for (PartitionSync &sync : syncs)
    if (sync.sharedGroup >= 0)
      barrierIds[sync.op.getOperation()] =
          sharedGroupIds.lookup(sync.sharedGroup);

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
    Value id = barrierIds.lookup(sync.op.getOperation());
    Region &partition = *sync.op->getParentRegion();
    Value waitPhase = resolvePartitionSyncPhase(
        sync.op, partition, dominance, rewriter, phase);
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.arrive.none.phaseid"),
        ValueRange{id});
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.wait"),
        ValueRange{id, waitPhase});
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

static LogicalResult lowerMultiStaticWarpSpecialize(
    LLVM::LLVMFuncOp func, ttg::WarpSpecializeOp ws,
    IRRewriter &rewriter) {
  SmallVector<Region *> partitionRegions(ws.getPartitionRegions().begin(),
                                         ws.getPartitionRegions().end());
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (ws.getNumResults() != 0 || partitionRegions.size() < 2 ||
      partitionRegions.size() != partitionNumWarps.size())
    return ws.emitOpError(
        "mthreads TLE multi-partition lowering requires at least two worker "
        "partitions, matching warp counts, and no results");
  if (llvm::any_of(partitionNumWarps,
                   [](int32_t count) { return count <= 0; }))
    return ws.emitOpError(
        "mthreads TLE multi-partition lowering requires positive worker warp "
        "counts");
  if (ws.getDefaultRegion().empty() ||
      llvm::any_of(partitionRegions,
                   [](Region *region) { return region->empty(); }))
    return ws.emitOpError("mthreads TLE static partitions must not be empty");

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  SmallVector<Block *> partitionRoots;
  SmallVector<ttg::WarpReturnOp> partitionReturns;
  for (Region *partition : partitionRegions) {
    Block &entry = partition->front();
    if (entry.getNumArguments() != captures.size())
      return ws.emitOpError(
          "worker capture count changed during multi-partition lowering");
    for (auto [argument, capture] : llvm::zip(entry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError(
            "worker capture types were not converted consistently");
      argument.replaceAllUsesWith(capture);
    }
    entry.eraseArguments([](BlockArgument) { return true; });
    partitionRoots.push_back(&entry);
    partition->walk(
        [&](ttg::WarpReturnOp op) { partitionReturns.push_back(op); });
  }

  Region &consumerRegion = ws.getDefaultRegion();
  SmallVector<ttg::WarpYieldOp> consumerYields;
  consumerRegion.walk(
      [&](ttg::WarpYieldOp op) { consumerYields.push_back(op); });
  if (partitionReturns.empty() || consumerYields.empty())
    return ws.emitOpError("static partitions lost their terminators");

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!consumerWarpsAttr || !threadsPerWarpAttr ||
      consumerWarpsAttr.getInt() <= 0 || threadsPerWarpAttr.getInt() <= 0)
    return ws.emitOpError("multi-partition lowering requires positive "
                          "ttg.num-warps and ttg.threads-per-warp");
  int64_t workerStartWarp = consumerWarpsAttr.getInt();
  int64_t threadsPerWarp = threadsPerWarpAttr.getInt();
  if (workerStartWarp >
      std::numeric_limits<int32_t>::max() / threadsPerWarp)
    return ws.emitOpError("static partition boundary exceeds int32 range");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  for (Region *partition : partitionRegions)
    funcBlocks.splice(continuation->getIterator(), partition->getBlocks());
  Block *consumerRoot = &consumerRegion.front();
  funcBlocks.splice(continuation->getIterator(), consumerRegion.getBlocks());

  for (ttg::WarpReturnOp op : partitionReturns) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }
  for (ttg::WarpYieldOp op : consumerYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "mthreads TLE static consumer must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value firstWorkerThread = arith::ConstantIntOp::create(
      rewriter, loc, workerStartWarp * threadsPerWarp, 32);
  Value isDefault = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, tid, firstWorkerThread);
  Block *workerDispatch =
      rewriter.createBlock(&funcBody, continuation->getIterator());
  rewriter.setInsertionPointToEnd(dispatch);
  cf::CondBranchOp::create(rewriter, loc, isDefault, consumerRoot, ValueRange{},
                           workerDispatch, ValueRange{});

  for (unsigned index = 0; index + 1 < partitionRegions.size(); ++index) {
    workerStartWarp += partitionNumWarps[index];
    if (workerStartWarp <= 0 ||
        workerStartWarp >
            std::numeric_limits<int32_t>::max() / threadsPerWarp) {
      emitError(loc, "mthreads TLE static partition boundary exceeds int32 "
                     "range");
      return failure();
    }
    int64_t upperThread = workerStartWarp * threadsPerWarp;
    rewriter.setInsertionPointToEnd(workerDispatch);
    Value upper =
        arith::ConstantIntOp::create(rewriter, loc, upperThread, 32);
    Value inPartition = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, tid, upper);
    Block *nextDispatch =
        rewriter.createBlock(&funcBody, continuation->getIterator());
    rewriter.setInsertionPointToEnd(workerDispatch);
    cf::CondBranchOp::create(rewriter, loc, inPartition,
                             partitionRoots[index], ValueRange{}, nextDispatch,
                             ValueRange{});
    workerDispatch = nextDispatch;
  }
  rewriter.setInsertionPointToEnd(workerDispatch);
  cf::BranchOp::create(rewriter, loc, partitionRoots.back());
  return success();
}

static LogicalResult lowerStaticWarpSpecialize(LLVM::LLVMFuncOp func,
                                               ttg::WarpSpecializeOp ws,
                                               IRRewriter &rewriter) {
  if (ws.getPartitionRegions().size() > 1)
    return lowerMultiStaticWarpSpecialize(func, ws, rewriter);
  if (ws.getNumResults() != 0 || ws.getPartitionRegions().size() != 1 ||
      ws.getPartitionNumWarps().size() != 1)
    return ws.emitOpError(
        "mthreads TLE late lowering requires one producer, one consumer, and "
        "no results");

  Region &producerRegion = *ws.getPartitionRegions().front();
  Region &consumerRegion = ws.getDefaultRegion();
  if (producerRegion.empty() || consumerRegion.empty())
    return ws.emitOpError("mthreads TLE static partitions must not be empty");

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  Block &producerEntry = producerRegion.front();
  if (producerEntry.getNumArguments() != captures.size())
    return ws.emitOpError("producer capture count changed during lowering");
  for (auto [argument, capture] :
       llvm::zip(producerEntry.getArguments(), captures)) {
    if (argument.getType() != capture.getType())
      return ws.emitOpError(
          "producer capture types were not converted consistently");
    argument.replaceAllUsesWith(capture);
  }
  producerEntry.eraseArguments([](BlockArgument) { return true; });

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!consumerWarpsAttr || !threadsPerWarpAttr ||
      consumerWarpsAttr.getInt() <= 0 || threadsPerWarpAttr.getInt() <= 0)
    return ws.emitOpError("late lowering requires positive ttg.num-warps and "
                          "ttg.threads-per-warp");
  if (ws.getPartitionNumWarps().front() <= 0)
    return ws.emitOpError(
        "late lowering requires a positive worker warp count");
  int64_t boundary64 = consumerWarpsAttr.getInt();
  if (boundary64 >
      std::numeric_limits<int32_t>::max() / threadsPerWarpAttr.getInt())
    return ws.emitOpError("static partition boundary exceeds int32 range");
  boundary64 *= threadsPerWarpAttr.getInt();
  int32_t boundary = static_cast<int32_t>(boundary64);

  // `default_producer` swaps the execution roles of the two regions.  The
  // frontend still reports `ttg.num-warps` for the default region and the
  // worker warp count for the partition region, so the warp boundary itself
  // does not move.  Only the dispatch predicates/roots need to be exchanged;
  // the producer join below remains useful to keep producer warps alive until
  // the consumer has retired its asynchronous work.
  bool defaultProducer = ws->hasAttr(kDefaultProducerAttr);

  // The frontend keeps the default region and worker region in fixed
  // positions, while `default_producer` swaps their execution roles.  Do not
  // infer the control-flow terminator from the physical region: a producer
  // may end in warp_yield (default region), and a consumer may end in
  // warp_return (worker region).  Rewriting by role avoids routing consumer
  // workers back through the producer join when the roles are swapped.
  Region &producerExecRegion = defaultProducer ? consumerRegion : producerRegion;
  Region &consumerExecRegion = defaultProducer ? producerRegion : consumerRegion;
  SmallVector<Operation *> producerTerminators;
  producerExecRegion.walk([&](Operation *op) {
    if (isa<ttg::WarpReturnOp, ttg::WarpYieldOp>(op))
      producerTerminators.push_back(op);
  });
  SmallVector<Operation *> consumerTerminators;
  consumerExecRegion.walk([&](Operation *op) {
    if (isa<ttg::WarpReturnOp, ttg::WarpYieldOp>(op))
      consumerTerminators.push_back(op);
  });
  if (producerTerminators.empty() || consumerTerminators.empty())
    return ws.emitOpError("static partitions lost their terminators");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  Block *producerRoot = defaultProducer ? &consumerRegion.front()
                                        : &producerRegion.front();
  Block *consumerRoot = defaultProducer ? &producerRegion.front()
                                        : &consumerRegion.front();
  funcBlocks.splice(continuation->getIterator(), producerRegion.getBlocks());
  Block *producerJoin =
      rewriter.createBlock(&funcBody, continuation->getIterator());
  funcBlocks.splice(continuation->getIterator(), consumerRegion.getBlocks());

  for (Operation *op : producerTerminators) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op->getLoc(), producerJoin);
    rewriter.eraseOp(op);
  }
  for (Operation *op : consumerTerminators) {
    if (auto yield = dyn_cast<ttg::WarpYieldOp>(op);
        yield && yield.getNumOperands() != 0)
      return op->emitOpError(
          "mthreads TLE static consumer must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op->getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value boundaryValue =
      arith::ConstantIntOp::create(rewriter, loc, boundary, 32);
  Value isProducer = arith::CmpIOp::create(
      rewriter, loc,
      defaultProducer ? arith::CmpIPredicate::ult
                      : arith::CmpIPredicate::uge,
      tid, boundaryValue);
  cf::CondBranchOp::create(rewriter, loc, isProducer, producerRoot,
                           ValueRange{}, producerJoin, ValueRange{});

  rewriter.setInsertionPointToEnd(producerJoin);
  Value isConsumer = arith::CmpIOp::create(
      rewriter, loc,
      defaultProducer ? arith::CmpIPredicate::uge
                      : arith::CmpIPredicate::ult,
      tid, boundaryValue);
  cf::CondBranchOp::create(rewriter, loc, isConsumer, consumerRoot,
                           ValueRange{}, continuation, ValueRange{});
  return success();
}

class LowerWarpSpecializePass
    : public impl::TritonMUSAGPUTLELowerWarpSpecializeBase<
          LowerWarpSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto stripExplicitCtaSyncMarkers = [&]() {
      module.walk([&](LLVM::CallIntrinsicOp call) {
        call->removeAttr(kExplicitCtaSyncAttr);
      });
    };
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty()) {
      stripExplicitCtaSyncMarkers();
      return;
    }
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "mthreads TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    auto func = marked.front()->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      marked.front().emitOpError(
          "mthreads TLE late lowering requires an LLVM function");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(lowerWarpGroupBarriers(func, marked.front(), rewriter)) ||
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter))) {
      signalPassFailure();
      return;
    }
    stripExplicitCtaSyncMarkers();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
