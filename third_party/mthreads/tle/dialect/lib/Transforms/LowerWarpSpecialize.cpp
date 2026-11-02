#ifdef __TLE__

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "MUSATLE/Transforms/TransformAttrs.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEPREPAREWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace musa_tle = triton::musa_tle;

static FailureOr<unsigned>
countExplicitCtaSyncs(Region &partition,
                      std::optional<unsigned> workerIndex = std::nullopt) {
  unsigned count = 0;
  bool invalid = false;
  partition.walk([&](ttg::BarrierOp barrier) {
    if (!barrier->hasAttr(musa_tle::kMUSATLEExplicitCtaSyncAttr))
      return WalkResult::advance();
    if (barrier.getAddrSpace() != ttg::AddrSpace::Local) {
      barrier.emitOpError(
          "MUSA TLE explicit CTA sync must target local memory");
      invalid = true;
      return WalkResult::interrupt();
    }
    if (barrier->getBlock() != &partition.front()) {
      auto diagnostic = barrier.emitOpError("MUSA TLE explicit CTA sync in ");
      if (workerIndex)
        diagnostic << "worker partition #" << *workerIndex;
      else
        diagnostic << "the default partition";
      diagnostic << " must be directly in the partition top-level block";
      invalid = true;
      return WalkResult::interrupt();
    }
    ++count;
    return WalkResult::advance();
  });
  if (invalid)
    return failure();
  return count;
}

static LogicalResult validateExplicitCtaSyncs(ttg::WarpSpecializeOp ws) {
  FailureOr<unsigned> defaultCount =
      countExplicitCtaSyncs(ws.getDefaultRegion());
  if (failed(defaultCount))
    return failure();

  for (auto [workerIndex, workerRegion] :
       llvm::enumerate(ws.getPartitionRegions())) {
    FailureOr<unsigned> workerCount =
        countExplicitCtaSyncs(*workerRegion, workerIndex);
    if (failed(workerCount))
      return failure();
    if (*workerCount != *defaultCount)
      return ws.emitOpError()
             << "MUSA TLE explicit CTA sync count mismatch: default "
                "partition has "
             << *defaultCount << " but worker partition #" << workerIndex
             << " has " << *workerCount;
  }
  return success();
}

static bool isAsyncSqmma(Operation *op) {
  if (auto squadDot = dyn_cast<triton::musa::SquadDotOp>(op))
    return squadDot.getIsAsync();
  if (auto sqmma = dyn_cast<triton::mtgpu::SqmmaOp>(op))
    return sqmma.getIsAsync();
  return false;
}

static bool containsAsyncSqmmaAndGlobalStore(scf::ForOp loop) {
  bool hasAsyncSqmma = false;
  bool hasGlobalStore = false;
  loop.walk([&](Operation *op) {
    hasAsyncSqmma |= isAsyncSqmma(op);
    hasGlobalStore |= isa<triton::StoreOp>(op);
  });
  return hasAsyncSqmma && hasGlobalStore;
}

static bool equivalentBarrierIds(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  APInt lhsConstant;
  APInt rhsConstant;
  return matchPattern(lhs, m_ConstantInt(&lhsConstant)) &&
         matchPattern(rhs, m_ConstantInt(&rhsConstant)) &&
         lhsConstant == rhsConstant;
}

static bool isConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

// Match the canonical alternating phase emitted by TLE named barriers:
//   current epoch: iteration & 1
//   next epoch:    (iteration + 1) & 1
static bool matchesEpochPhase(Value phase, Value iteration, bool nextEpoch) {
  auto andOp = phase.getDefiningOp<arith::AndIOp>();
  if (!andOp)
    return false;

  Value epoch;
  if (isConstantInt(andOp.getLhs(), 1))
    epoch = andOp.getRhs();
  else if (isConstantInt(andOp.getRhs(), 1))
    epoch = andOp.getLhs();
  else
    return false;

  if (!nextEpoch)
    return epoch == iteration;
  auto addOp = epoch.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return false;
  return (addOp.getLhs() == iteration && isConstantInt(addOp.getRhs(), 1)) ||
         (addOp.getRhs() == iteration && isConstantInt(addOp.getLhs(), 1));
}

// A persistent consumer may already carry an explicit epoch barrier, for
// example an outer-loop header wait paired with a latch arrival.  Only inspect
// operations directly in the epoch loop body: pipe full/empty barriers live in
// the nested K loop and must not suppress the compiler-provided rendezvous.
static FailureOr<std::optional<Value>>
findTopLevelEpochRendezvous(scf::ForOp epochLoop) {
  SmallVector<triton::musa::WaitBarrierOp> waits;
  SmallVector<triton::musa::WarpArriveBarrierOp> arrivals;
  for (Operation &op : epochLoop.getBody()->getOperations()) {
    if (auto wait = dyn_cast<triton::musa::WaitBarrierOp>(op))
      waits.push_back(wait);
    if (auto arrive = dyn_cast<triton::musa::WarpArriveBarrierOp>(op))
      arrivals.push_back(arrive);
  }
  std::optional<Value> matchedBarrier;
  for (triton::musa::WaitBarrierOp wait : waits) {
    for (triton::musa::WarpArriveBarrierOp arrive : arrivals) {
      if (!equivalentBarrierIds(wait.getBarId(), arrive.getBarId()))
        continue;
      // The supported explicit protocol waits for the next phase at the loop
      // header and arrives on the current phase at the latch.  This is the
      // same alternating protocol used by compiler-generated epoch syncs.
      if (!wait->isBeforeInBlock(arrive) ||
          !matchesEpochPhase(wait.getPhaseId(), epochLoop.getInductionVar(),
                             /*nextEpoch=*/true) ||
          !matchesEpochPhase(arrive.getPhaseId(), epochLoop.getInductionVar(),
                             /*nextEpoch=*/false)) {
        epochLoop.emitOpError(
            "explicit consumer epoch barrier must wait on "
            "(iteration + 1) & 1 at the loop header and arrive on "
            "iteration & 1 at the loop latch");
        return failure();
      }
      if (matchedBarrier) {
        epochLoop.emitOpError(
            "requires exactly one explicit consumer epoch barrier pair");
        return failure();
      }
      matchedBarrier = wait.getBarId();
    }
  }
  return matchedBarrier;
}

static LogicalResult validateExplicitEpochBarrierInitialization(
    ttg::WarpSpecializeOp ws, Value barrier, int32_t consumerWarps) {
  auto func = ws->getParentOfType<triton::FuncOp>();
  SmallVector<triton::musa::InitArrivalOp> matchingInitializers;
  func.walk([&](triton::musa::InitArrivalOp init) {
    if (equivalentBarrierIds(init.getBarId(), barrier))
      matchingInitializers.push_back(init);
  });
  if (matchingInitializers.size() != 1)
    return ws.emitOpError(
        "explicit consumer epoch barrier requires exactly one matching "
        "initialization");

  triton::musa::InitArrivalOp init = matchingInitializers.front();
  APInt arriveCount;
  if (!matchPattern(init.getArriveCount(), m_ConstantInt(&arriveCount)) ||
      arriveCount.getSExtValue() != consumerWarps)
    return ws.emitOpError()
           << "explicit consumer epoch barrier arrival count must equal the "
              "total consumer warp count ("
           << consumerWarps << ")";
  if (!isConstantInt(init.getPhaseId(), 0))
    return ws.emitOpError(
        "explicit consumer epoch barrier must initialize phase 0");
  return success();
}

static FailureOr<int64_t> getPositiveConstantStep(scf::ForOp loop) {
  APInt step;
  if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      step.getSExtValue() <= 0)
    return failure();
  return step.getSExtValue();
}

// Worker partitions are isolated from above, so a captured loop bound is a
// partition block argument there and the original SSA value in the default
// partition.  Compare the common captured value rather than those two local
// spellings when validating that every consumer executes the same epochs.
static Value canonicalConsumerBound(Value value, Region &consumer,
                                    ttg::WarpSpecializeOp ws) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &consumer.front())
    return value;
  unsigned index = argument.getArgNumber();
  ValueRange captures = ws.getPartitionOp().getExplicitCaptures();
  if (index >= captures.size())
    return value;
  return captures[index];
}

static bool equivalentConsumerBounds(Value lhs, Region &lhsConsumer, Value rhs,
                                     Region &rhsConsumer,
                                     ttg::WarpSpecializeOp ws) {
  lhs = canonicalConsumerBound(lhs, lhsConsumer, ws);
  rhs = canonicalConsumerBound(rhs, rhsConsumer, ws);
  if (lhs == rhs)
    return true;

  APInt lhsConstant;
  APInt rhsConstant;
  return matchPattern(lhs, m_ConstantInt(&lhsConstant)) &&
         matchPattern(rhs, m_ConstantInt(&rhsConstant)) &&
         lhsConstant == rhsConstant;
}

// SQMMA wait-all retires issued work, but it does not reconverge the consumer
// warps or provide a completion edge between independent consumer partitions.
// Persistent kernels that reuse a warp-specialized CTA across output tiles
// therefore need one rendezvous at each consumer outer-loop latch. Bounds may
// be runtime values (the canonical persistent schedule is
// range(pid, num_tiles, NUM_SMS)), but every consumer must execute one
// unconditional top-level loop with equivalent lower/upper bounds and the
// same positive constant step. Late lowering
// merges equal markers into one hardware barrier and derives its alternating
// phase from the loop iteration count.
static LogicalResult markConsumerEpochRendezvous(ttg::WarpSpecializeOp ws,
                                                 int32_t consumerWarps,
                                                 IRRewriter &rewriter) {
  if (!ws->hasAttr(musa_tle::kMUSATLEConsumerEpochRendezvousRequiredAttr))
    return success();
  auto skipInference = [&]() {
    ws->removeAttr(musa_tle::kMUSATLEConsumerEpochRendezvousRequiredAttr);
    return success();
  };

  SmallVector<Region *> consumers{&ws.getDefaultRegion()};
  SmallVector<Region *> partitions(ws.getPartitionRegions().begin(),
                                   ws.getPartitionRegions().end());
  // The regular static layout places the producer in the final worker
  // partition.  The default region and preceding workers are consumers.
  for (unsigned index = 0; index + 1 < partitions.size(); ++index)
    consumers.push_back(partitions[index]);

  SmallVector<SmallVector<scf::ForOp>> candidatesByConsumer;
  candidatesByConsumer.reserve(consumers.size());
  bool hasPersistentCandidate = false;
  for (Region *consumer : consumers) {
    auto &candidates = candidatesByConsumer.emplace_back();
    consumer->walk([&](scf::ForOp loop) {
      if (containsAsyncSqmmaAndGlobalStore(loop))
        candidates.push_back(loop);
    });
    hasPersistentCandidate |= !candidates.empty();
  }
  if (!hasPersistentCandidate)
    // One-shot named readers have no outer epoch to rendezvous, while cyclic
    // non-SQMMA readers are already protected by the pipe's full/empty
    // barriers.  Only validate epoch structure after finding an actual
    // SQMMA+store loop candidate.
    return skipInference();

  SmallVector<scf::ForOp> epochLoops;
  SmallVector<std::optional<Value>> explicitEpochBarriers;
  Region *commonConsumer = nullptr;
  Value commonLower;
  Value commonUpper;
  std::optional<int64_t> commonStep;
  for (auto [consumer, candidates] :
       llvm::zip(consumers, candidatesByConsumer)) {
    Region *consumerRegion = consumer;
    if (llvm::any_of(candidates, [&](scf::ForOp loop) {
          return loop->getParentRegion() != consumerRegion;
        })) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires the consumer "
          "epoch loop to be an unconditional top-level operation in every "
          "consumer partition");
      return failure();
    }
    // Nested K loops contain SQMMA but no output store.  Requiring exactly
    // one top-level SQMMA+store loop keeps this inference deliberately
    // conservative and guarantees that every consumer reaches the shared
    // barrier once per epoch.
    if (candidates.size() != 1) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires exactly one "
          "unconditional top-level SQMMA/store epoch loop in every consumer "
          "partition");
      return failure();
    }
    scf::ForOp loop = candidates.front();
    FailureOr<int64_t> step = getPositiveConstantStep(loop);
    if (failed(step)) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires a positive "
          "constant consumer epoch loop step; use "
          "range(pid, num_tiles, NUM_SMS)");
      return failure();
    }
    if (commonConsumer &&
        (!equivalentConsumerBounds(commonLower, *commonConsumer,
                                   loop.getLowerBound(), *consumer, ws) ||
         !equivalentConsumerBounds(commonUpper, *commonConsumer,
                                   loop.getUpperBound(), *consumer, ws) ||
         *commonStep != *step)) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires equivalent "
          "lower and upper bounds and the same positive constant step in "
          "every consumer partition");
      return failure();
    }
    if (!commonConsumer) {
      commonConsumer = consumer;
      commonLower = loop.getLowerBound();
      commonUpper = loop.getUpperBound();
      commonStep = *step;
    }
    epochLoops.push_back(loop);
    FailureOr<std::optional<Value>> explicitEpoch =
        findTopLevelEpochRendezvous(loop);
    if (failed(explicitEpoch))
      return failure();
    explicitEpochBarriers.push_back(*explicitEpoch);
  }

  bool hasAnyExplicitEpoch = llvm::any_of(
      explicitEpochBarriers,
      [](const std::optional<Value> &barrier) { return barrier.has_value(); });
  bool hasAllExplicitEpoch = llvm::all_of(
      explicitEpochBarriers,
      [](const std::optional<Value> &barrier) { return barrier.has_value(); });
  if (hasAnyExplicitEpoch != hasAllExplicitEpoch)
    return ws.emitOpError(
        "MUSA TLE persistent named-reader consumers must either all provide "
        "an explicit top-level epoch wait/arrival pair or all rely on the "
        "compiler-provided rendezvous");
  if (hasAllExplicitEpoch) {
    Value commonBarrier = *explicitEpochBarriers.front();
    Region *commonBarrierConsumer = consumers.front();
    for (auto [consumer, barrier] :
         llvm::zip(ArrayRef<Region *>(consumers).drop_front(),
                   ArrayRef<std::optional<Value>>(explicitEpochBarriers)
                       .drop_front())) {
      if (!equivalentConsumerBounds(commonBarrier, *commonBarrierConsumer,
                                    *barrier, *consumer, ws))
        return ws.emitOpError(
            "MUSA TLE persistent named-reader consumers must use the same "
            "explicit top-level epoch barrier");
    }
    if (failed(validateExplicitEpochBarrierInitialization(ws, commonBarrier,
                                                          consumerWarps)))
      return failure();
    return skipInference();
  }

  for (scf::ForOp loop : epochLoops) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    rewriter.setInsertionPoint(yield);
    auto barrier =
        ttg::BarrierOp::create(rewriter, yield.getLoc(), ttg::AddrSpace::Local);
    barrier->setAttr(musa_tle::kMUSATLEConsumerEpochSyncAttr,
                     rewriter.getI32IntegerAttr(0));
  }
  ws->removeAttr(musa_tle::kMUSATLEConsumerEpochRendezvousRequiredAttr);
  return success();
}

class PrepareWarpSpecializePass
    : public impl::TritonMUSAGPUTLEPrepareWarpSpecializeBase<
          PrepareWarpSpecializePass> {
  LogicalResult prepareWarpSpecialize(ModuleOp mod, ttg::WarpSpecializeOp ws,
                                      IRRewriter &rewriter) {
    auto func = ws->getParentOfType<triton::FuncOp>();
    if (!func)
      return ws.emitOpError("requires a parent Triton function");

    Block &entry = func.getBody().front();
    if (ws->getBlock() != &entry)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize must be in the function entry "
          "block");
    if (!ws->getNextNode() || !isa<triton::ReturnOp>(ws->getNextNode()) ||
        ws->getNextNode()->getNextNode())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize must be the final operation "
          "before tt.return");
    if (ws.getNumResults() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize does not support results");

    auto partitions = ws.getPartitionOp();
    auto workerRegions = partitions.getPartitionRegions();
    ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
    if (workerRegions.empty())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires at least one worker "
          "partition");
    if (workerRegions.size() != workerWarps.size())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize worker region and warp-count "
          "sizes must match");
    if (auto requestedRegisters = ws.getRequestedRegisters()) {
      if (requestedRegisters->size() != workerRegions.size())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize requested-register count must "
            "match the worker partition count");
    }

    Region &defaultRegion = ws.getDefaultRegion();
    if (!defaultRegion.hasOneBlock())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires a single-block default "
          "region");

    Block &defaultBlock = defaultRegion.front();
    auto defaultYield =
        dyn_cast<ttg::WarpYieldOp>(defaultBlock.getTerminator());
    if (!defaultYield || defaultYield.getNumOperands() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default region must yield no "
          "values");

    ValueRange captures = partitions.getExplicitCaptures();
    for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
      if (!workerRegion.hasOneBlock())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must be single-block";
      Block &workerBlock = workerRegion.front();
      if (!isa<ttg::WarpReturnOp>(workerBlock.getTerminator()))
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must end with ttg.warp_return";
      if (workerBlock.getNumArguments() != captures.size())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " capture count mismatch";
      for (auto [captureIndex, argumentAndCapture] :
           llvm::enumerate(llvm::zip(workerBlock.getArguments(), captures))) {
        auto [argument, capture] = argumentAndCapture;
        if (argument.getType() != capture.getType())
          return ws.emitOpError()
                 << "MUSA TLE static warp_specialize worker partition #"
                 << workerIndex << " capture #" << captureIndex
                 << " type mismatch";
      }
    }
    DominanceInfo dominance(func);
    for (Value capture : captures) {
      if (!dominance.dominates(capture, ws.getOperation()))
        return ws.emitOpError(
            "MUSA TLE static warp_specialize capture must dominate the "
            "partition split");
    }

    if (failed(validateExplicitCtaSyncs(ws)))
      return failure();

    int32_t baseNumWarps = ttg::lookupNumWarps(ws);
    if (baseNumWarps <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default warp count must be "
          "positive");
    int32_t warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    if (warpSize <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize warp size must be positive");

    int64_t nextWarp = baseNumWarps;
    SmallVector<int32_t> workerStartIds;
    workerStartIds.reserve(workerWarps.size());
    for (auto [workerIndex, numWarps] : llvm::enumerate(workerWarps)) {
      if (numWarps <= 0)
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " requires a positive static warp count";
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
      workerStartIds.push_back(static_cast<int32_t>(nextWarp));
      nextWarp += numWarps;
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
    }
    if (nextWarp > std::numeric_limits<int32_t>::max() / warpSize)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize thread count exceeds int32");

    int32_t totalNumWarps = static_cast<int32_t>(nextWarp);
    if (auto existingStartIds = ws.getWarpGroupStartIds()) {
      if (existingStartIds->size() != workerStartIds.size() ||
          !llvm::equal(*existingStartIds, workerStartIds))
        return ws.emitOpError(
            "MUSA TLE static worker warp ranges must follow declaration "
            "order after the default partition");
    } else {
      ws.setWarpGroupStartIds(workerStartIds);
    }
    if (auto existing =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
      if (existing.getInt() != totalNumWarps)
        return ws.emitOpError(
            "MUSA TLE static warp_specialize conflicts with existing "
            "ttg.total-num-warps");
    } else {
      mod->setAttr("ttg.total-num-warps",
                   rewriter.getI32IntegerAttr(totalNumWarps));
    }

    int32_t consumerWarps = baseNumWarps;
    for (unsigned index = 0; index + 1 < workerWarps.size(); ++index)
      consumerWarps += workerWarps[index];
    if (failed(markConsumerEpochRendezvous(ws, consumerWarps, rewriter)))
      return failure();

    return success();
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(musa_tle::kMUSATLEStaticWarpSpecializeAttr))
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

    IRRewriter rewriter(&getContext());
    if (failed(prepareWarpSpecialize(mod, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
