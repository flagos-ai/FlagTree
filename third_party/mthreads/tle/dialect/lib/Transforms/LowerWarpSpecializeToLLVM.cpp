#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
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

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";

struct PartitionSync {
  LLVM::CallIntrinsicOp op;
  int32_t numWarps;
  Region *partition;
};

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

  SmallVector<PartitionSync> syncs;
  auto collect = [&](Region &region, int32_t warps) {
    region.walk([&](LLVM::CallIntrinsicOp call) {
      if (call.getIntrin() == kLocalSyncIntrinsic)
        syncs.push_back({call, warps, &region});
    });
  };
  collect(ws.getDefaultRegion(), defaultWarps);
  for (auto [region, warps] :
       llvm::zip_equal(ws.getPartitionRegions(), ws.getPartitionNumWarps()))
    collect(*region, warps);

  llvm::DenseSet<Region *> partitions;
  for (auto &sync : syncs)
    partitions.insert(sync.partition);
  if (syncs.empty())
    return success();

  auto reserved = ttmg::reserveBarrierIdRange(
      syncs.front().op, static_cast<int32_t>(partitions.size()));
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
  llvm::DenseMap<Region *, Value> barrierIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (PartitionSync &sync : syncs) {
    if (barrierIds.count(sync.partition))
      continue;
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, sync.numWarps, 32);
    initializationArgs.push_back({id, count});
    barrierIds[sync.partition] = id;
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
    Value id = barrierIds.lookup(sync.partition);
    // A partition reuses this resource at every rendezvous. Use the phase
    // returned by arrival, not a fixed phase that only works on the first use.
    Value arrived =
        LLVM::CallIntrinsicOp::create(
            rewriter, sync.op.getLoc(), rewriter.getI32Type(),
            rewriter.getStringAttr("llvm.musa.async.arrive"), ValueRange{id})
            .getResult(0);
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.wait"),
        ValueRange{id, arrived});
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
    Block *workerJoin =
        rewriter.createBlock(&funcBody, continuation->getIterator());
    funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());

    for (ttg::WarpReturnOp op : workerReturns.front()) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op.getLoc(), workerJoin);
      rewriter.eraseOp(op);
    }
    for (ttg::WarpYieldOp op : defaultYields) {
      if (op.getNumOperands() != 0)
        return op.emitOpError(
            "MUSA TLE static default partition must not yield values");
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op.getLoc(), continuation);
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
    Value isWorker = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::uge, tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isWorker, workerRoots.front(),
                             ValueRange{}, workerJoin, ValueRange{});

    rewriter.setInsertionPointToEnd(workerJoin);
    Value isDefault = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isDefault, defaultRoot,
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
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty())
      return;
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
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
