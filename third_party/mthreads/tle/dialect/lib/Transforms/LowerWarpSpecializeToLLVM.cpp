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
};

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

  Region &producerRegion = *ws.getPartitionRegions().front();
  Region &consumerRegion = ws.getDefaultRegion();
  SmallVector<LLVM::CallIntrinsicOp> redundantSyncs;
  SmallVector<PartitionSync> syncs;

  producerRegion.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() == kLocalSyncIntrinsic)
      redundantSyncs.push_back(call);
  });

  bool seenSqmma = false;
  consumerRegion.walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
    if (!call)
      return WalkResult::advance();
    if (call.getIntrin().starts_with("llvm.musa.sqmma.fmma.")) {
      seenSqmma = true;
      return WalkResult::advance();
    }
    if (call.getIntrin() != kLocalSyncIntrinsic)
      return WalkResult::advance();
    if (seenSqmma)
      syncs.push_back({call, consumerWarps});
    else
      redundantSyncs.push_back(call);
    return WalkResult::advance();
  });

  for (LLVM::CallIntrinsicOp redundant : redundantSyncs)
    rewriter.eraseOp(redundant);
  if (syncs.empty())
    return success();

  auto reserved = ttmg::reserveBarrierIdRange(
      syncs.front().op, static_cast<int32_t>(syncs.size()));
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
  llvm::DenseMap<Operation *, Value> barrierIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (PartitionSync &sync : syncs) {
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, sync.numWarps, 32);
    initializationArgs.push_back({id, count});
    barrierIds[sync.op.getOperation()] = id;
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
    Value id = barrierIds.lookup(sync.op.getOperation());
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.arrive.none.phaseid"),
        ValueRange{id});
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.wait"), ValueRange{id, phase});
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
  int64_t boundary64 = consumerWarpsAttr.getInt() * threadsPerWarpAttr.getInt();
  if (boundary64 > std::numeric_limits<int32_t>::max())
    return ws.emitOpError("static partition boundary exceeds int32 range");
  int32_t boundary = static_cast<int32_t>(boundary64);

  SmallVector<ttg::WarpReturnOp> producerReturns;
  producerRegion.walk(
      [&](ttg::WarpReturnOp op) { producerReturns.push_back(op); });
  SmallVector<ttg::WarpYieldOp> consumerYields;
  consumerRegion.walk(
      [&](ttg::WarpYieldOp op) { consumerYields.push_back(op); });
  if (producerReturns.empty() || consumerYields.empty())
    return ws.emitOpError("static partitions lost their terminators");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  Block *producerRoot = &producerRegion.front();
  Block *consumerRoot = &consumerRegion.front();
  funcBlocks.splice(continuation->getIterator(), producerRegion.getBlocks());
  Block *producerJoin =
      rewriter.createBlock(&funcBody, continuation->getIterator());
  funcBlocks.splice(continuation->getIterator(), consumerRegion.getBlocks());

  for (ttg::WarpReturnOp op : producerReturns) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), producerJoin);
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
  Value boundaryValue =
      arith::ConstantIntOp::create(rewriter, loc, boundary, 32);
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
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
