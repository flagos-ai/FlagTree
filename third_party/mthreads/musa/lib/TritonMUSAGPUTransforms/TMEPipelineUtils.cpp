#include "TMEPipelineUtils.h"

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::musa::pipeline {
namespace {

static void
allocTMABuffers(scf::ForOp forOp,
                llvm::MapVector<Operation *, Value> &tmaBufferMapping,
                int maxStage) {
  IRRewriter rewriter(forOp);
  forOp.walk([&](tt::MakeTensorDescOp op) {
    auto loc = op.getLoc();
    Value alloc = ttg::GlobalScratchAllocOp::create(
        rewriter, loc, triton::getPointerType(rewriter.getI8Type()),
        maxStage * ttng::TMA_SIZE_BYTES, ttng::TMA_ALIGN);
    tmaBufferMapping[op.getOperation()] = alloc;
  });
}

static Value subviewTMADescriptor(OpBuilder &builder, Location loc, Value alloc,
                                  Value counter) {
  Value tmaSizeVal =
      arith::ConstantIntOp::create(builder, loc, ttng::TMA_SIZE_BYTES, 32);
  Value offset = arith::MulIOp::create(builder, loc, tmaSizeVal, counter);
  return tt::AddPtrOp::create(builder, loc, alloc.getType(), alloc, offset);
}

static LogicalResult rewriteTMABufferUpdates(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, Value> &tmaBufferMapping,
    ArrayRef<BlockArgument> tmaCounters, int numBuffers, Value one, Value zero,
    tt::CoarseSchedule &schedule) {
  assert(tmaBufferMapping.size() == tmaCounters.size());

  OpBuilder auxBuilder(forOp);
  Value numBuffersVal =
      arith::ConstantIntOp::create(auxBuilder, forOp.getLoc(), numBuffers, 32);

  for (auto [iOp, pair] : llvm::enumerate(tmaBufferMapping)) {
    auto &[op, alloc] = pair;
    auto makeDescOp = cast<tt::MakeTensorDescOp>(op);

    tt::OpBuilderForStage builder(makeDescOp.getLoc(), makeDescOp, schedule);
    BlockArgument counter = tmaCounters[iOp];
    Value nextBuf =
        subviewTMADescriptor(builder, builder.getLoc(), alloc, counter);
    if (failed(ttng::createTMADesc(nextBuf, makeDescOp, builder)))
      return failure();
    ttng::TensormapFenceproxyAcquireOp::create(builder, nextBuf);
    Value nextDesc = ttng::ReinterpretTensorDescOp::create(
        builder, makeDescOp.getType(), nextBuf);

    makeDescOp.getResult().replaceAllUsesWith(nextDesc);

    Value nextCounter = createIncrementModulo(
        builder, builder.getLoc(), counter, numBuffersVal, zero, one);

    IRRewriter rewriter(forOp);
    nextCounter = triton::sinkValueRedefinition(rewriter, counter, nextCounter,
                                                op->getBlock());

    auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    forYield.setOperand(counter.getArgNumber() - 1, nextCounter);
    makeDescOp.erase();
  }
  return success();
}

struct TMEStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TMEStore> getTMEStores(scf::ForOp forOp) {
  SmallVector<TMEStore> tmaStores;
  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::DescriptorStoreLikeOpInterface>(op)) {
      tmaStores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return tmaStores;
}

static FailureOr<Value> createStoreAlloc(scf::ForOp &forOp,
                                         const TMEStore &store) {
  OpBuilder builder(forOp);
  auto memDescTy = triton::musa::resolveDescriptorStoreLandingMemDescType(
      cast<tt::DescriptorStoreLikeOpInterface>(store.op),
      /*mutableMemory=*/true);
  if (failed(memDescTy)) {
    store.op->emitOpError("pipelined descriptor store requires normalized "
                          "canonical landing memdesc encoding");
    return failure();
  }
  return ttg::LocalAllocOp::create(builder, store.op->getLoc(), *memDescTy)
      .getResult();
}

static LogicalResult createMUSATMEStoreAsyncCopy(const TMEStore &store,
                                                 Value alloc) {
  OpBuilder builder(store.op);
  Location loc = store.op->getLoc();

  auto descBlockTy = store.desc.getType().getSignlessBlockType();
  auto coord = triton::musa::materializeTMECoordValues(
      loc, cast<tt::DescriptorStoreOp>(store.op).getIndices(), builder);
  if (failed(coord)) {
    store.op->emitOpError("unable to materialize pipelined TME store block "
                          "info");
    return failure();
  }
  auto issueMemDescTy = triton::musa::resolveDescriptorStoreLandingMemDescType(
      cast<tt::DescriptorStoreLikeOpInterface>(store.op),
      /*mutableMemory=*/false);
  if (failed(issueMemDescTy)) {
    store.op->emitOpError("pipelined descriptor store requires normalized "
                          "immutable landing memdesc encoding");
    return failure();
  }
  Value issueAlloc = alloc;
  auto allocTy = cast<ttg::MemDescType>(alloc.getType());
  if (allocTy != *issueMemDescTy) {
    issueAlloc =
        ttg::MemDescReinterpretOp::create(builder, loc, *issueMemDescTy, alloc);
  }
  auto config = triton::musa::resolveFinalTMECopyConfig(
      *issueMemDescTy, descBlockTy.getShape(),
      triton::musa::TMECopyKind::LocalToGlobal);
  if (failed(config)) {
    store.op->emitOpError("unable to resolve pipelined TME store config");
    return failure();
  }

  Value pred = arith::ConstantIntOp::create(builder, loc, 1, 1);

  triton::musa::TMEStoreReadWaitOp::create(builder, loc);
  ttg::LocalStoreOp::create(builder, loc, store.src, alloc);
  triton::musa::createAsyncTMECopyLocalToGlobal(
      builder, loc, store.desc, *coord, issueAlloc, pred, *config);
  triton::musa::TMEStoreCommitOp::create(builder, loc);

  store.op->erase();
  return success();
}

static void lowerTMADescriptorCreation(scf::ForOp forOp) {
  tt::CoarseSchedule schedule(3);
  (void)mlir::triton::musa::pipeline::lowerTMADescriptors(forOp, schedule);
}

} // namespace

scf::ForOp lowerTMADescriptors(scf::ForOp forOp, tt::CoarseSchedule &schedule) {
  llvm::MapVector<Operation *, Value> tmaBufferMapping;
  int maxStage = schedule.getNumStages() - 1;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<ttng::WarpGroupDotOp>(&op)) {
      maxStage += 1;
      break;
    }
  }
  allocTMABuffers(forOp, tmaBufferMapping, maxStage);
  if (tmaBufferMapping.empty())
    return forOp;

  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  unsigned tmaCounterArgsStartIdx = newOperandIndex + newOperands.size();
  for (int i = 0; i < static_cast<int>(tmaBufferMapping.size()); ++i)
    newOperands.push_back(zero);

  forOp = addIterArgsToLoop(builder, forOp, newOperands);

  auto tmaCounters = ArrayRef<BlockArgument>(forOp.getBody()->getArguments())
                         .slice(tmaCounterArgsStartIdx);

  auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i)
    forYield.getResultsMutable().append(newOperands[i]);

  if (failed(rewriteTMABufferUpdates(forOp, tmaBufferMapping, tmaCounters,
                                     maxStage, one, zero, schedule))) {
    llvm::report_fatal_error("Failed to rewrite MUSA TMA descriptor updates");
  }
  return forOp;
}

FailureOr<bool> pipelineTMEStores(scf::ForOp forOp) {
  SmallVector<TMEStore> tmaStores = getTMEStores(forOp);
  if (tmaStores.empty())
    return false;

  struct StoreAllocEntry {
    ttg::MemDescType memDescTy;
    Value alloc;
  };

  DenseMap<Operation *, Value> storeToAlloc;
  SmallVector<StoreAllocEntry> allocs;
  for (const TMEStore &store : tmaStores) {
    if (!isa<tt::DescriptorStoreOp>(store.op)) {
      store.op->emitOpError("pipelined descriptor scatter/reduce is not "
                            "supported on MUSA");
      return failure();
    }
    auto allocMemDescTy =
        triton::musa::resolveDescriptorStoreLandingMemDescType(
            cast<tt::DescriptorStoreLikeOpInterface>(store.op),
            /*mutableMemory=*/true);
    if (failed(allocMemDescTy)) {
      store.op->emitOpError("pipelined descriptor store requires normalized "
                            "canonical landing memdesc encoding");
      return failure();
    }
    Value alloc;
    for (const StoreAllocEntry &entry : allocs) {
      if (entry.memDescTy == *allocMemDescTy) {
        alloc = entry.alloc;
        break;
      }
    }
    if (!alloc) {
      auto createdAlloc = createStoreAlloc(forOp, store);
      if (failed(createdAlloc))
        return failure();
      alloc = *createdAlloc;
      allocs.push_back(StoreAllocEntry{*allocMemDescTy, alloc});
    }
    storeToAlloc[store.op] = alloc;
  }

  bool hasDeviceSideTMA = llvm::any_of(tmaStores, [](const TMEStore &store) {
    return !triton::isHostSideDescriptor(store.desc);
  });
  for (const TMEStore &store : tmaStores) {
    if (failed(createMUSATMEStoreAsyncCopy(store, storeToAlloc[store.op])))
      return failure();
  }

  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  triton::musa::TMEStoreReadWaitOp::create(builder, forOp->getLoc());
  for (const StoreAllocEntry &entry : allocs)
    ttg::LocalDeallocOp::create(builder, forOp->getLoc(), entry.alloc);

  if (hasDeviceSideTMA)
    lowerTMADescriptorCreation(forOp);
  return true;
}

} // namespace mlir::triton::musa::pipeline
