//===- ConvertWarpSpecializeToLLVM.cpp - Iluvatar WS lowering -------------===//
//
// Ported from third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/
//     ConvertWarpSpecializeToLLVM.cpp
//
// This pass lowers `ttg.warp_specialize` into a warp-group dispatch loop at the
// LLVM level, exactly like the NVIDIA backend.
//
//===----------------------------------------------------------------------===//
// [WA] Iluvatar ivcore11 hardware limitations
//
// ivcore11 has no usable setmaxnreg in this pipeline. WS lowering therefore
// keeps register reallocation as a no-op and uses the backend software barrier
// protocol for warp-group synchronization.
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "Utility.h"
#include "TritonILUVATARGPUToLLVM/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <numeric>

namespace mlir::triton {
#define GEN_PASS_DEF_ILUVATARWARPSPECIALIZETOLLVM
#include "TritonILUVATARGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::triton::ILUVATAR::TargetInfo;

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

static void convertOpTypes(Operation *op, const TypeConverter &typeConverter) {
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  for (Value &operand : operands) {
    Type type = typeConverter.convertType(operand.getType());
    if (type != operand.getType()) {
      operand =
          UnrealizedConversionCastOp::create(b, type, operand).getResult(0);
    }
  }
  op->setOperands(operands);

  for (Region &region : op->getRegions()) {
    b.setInsertionPointToStart(&region.front());
    for (BlockArgument arg : llvm::to_vector(region.getArguments())) {
      Type type = typeConverter.convertType(arg.getType());
      BlockArgument newArg = region.addArgument(type, arg.getLoc());
      auto cast = UnrealizedConversionCastOp::create(b, arg.getType(), newArg);
      arg.replaceAllUsesWith(cast.getResult(0));
      region.eraseArgument(0);
    }
  }

  SmallVector<Type> resultTypes;
  (void)typeConverter.convertTypes(op->getResultTypes(), resultTypes);
  if (TypeRange(resultTypes) == op->getResultTypes())
    return;
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       resultTypes, op->getAttrs());
  for (Region &region : op->getRegions())
    state.addRegion()->takeBody(region);
  b.setInsertionPoint(op);
  Operation *newOp = b.create(state);

  SmallVector<Value> results;
  for (auto [i, result, type] :
       llvm::enumerate(newOp->getResults(), op->getResultTypes())) {
    auto cast = UnrealizedConversionCastOp::create(b, type, result);
    op->getResult(i).replaceAllUsesWith(cast.getResult(0));
  }
  op->erase();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Reserve one barrier for the default warp group, one for the start barrier,
// and one for the end barrier.
enum BarrierIndex {
  kDefaultWarpGroupBarrierIdx,
  kSwitchLoopBarrierIdx,

  kNumReservedBarriers,
  kNumBarriers = 16
};

static constexpr char kNamedBarStateName[] = "__ws_namedbar_state";

// A counter slot can be reused by more than one warp-specialize region.  The
// fixed-width target form is valid only when every barrier using that slot
// has the same participant warp count; otherwise use the packed generation
// protocol below.
using BarrierWarpCounts = llvm::DenseMap<unsigned, unsigned>;

static void recordBarrierWarpCount(BarrierWarpCounts &counts,
                                   unsigned barIdx, unsigned numThreads,
                                   unsigned threadsPerWarp) {
  unsigned numWarps = numThreads / threadsPerWarp;
  auto [it, inserted] = counts.try_emplace(barIdx, numWarps);
  if (!inserted && it->second != numWarps)
    it->second = 0;
}

static LLVM::GlobalOp getOrCreateSwBarrierState(ModuleOp mod,
                                                unsigned numBarriers) {
  if (auto g = dyn_cast_or_null<LLVM::GlobalOp>(
          mod.lookupSymbol(kNamedBarStateName)))
    return g;
  OpBuilder rewriter(mod.getBodyRegion());
  auto arrTy = LLVM::LLVMArrayType::get(i32_ty, numBarriers);
  return LLVM::GlobalOp::create(
      rewriter, mod.getLoc(), arrTy, /*isConstant=*/false,
      LLVM::Linkage::Internal, kNamedBarStateName, /*value=*/Attribute(),
      /*alignment=*/4, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
}

// Synchronize exactly `numThreads` threads at the native named barrier slot.
static void createNamedBarrier(RewriterBase &rewriter, Location loc,
                                 LLVM::GlobalOp state,
                                 unsigned threadsPerWarp, unsigned barIdx,
                                 unsigned numThreads,
                                 bool useFixedWarpCount) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  assert(threadsPerWarp > 0 && numThreads % threadsPerWarp == 0 &&
         "warp-group size must be a multiple of threadsPerWarp");
  unsigned numWarps = numThreads / threadsPerWarp;
  MLIRContext *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
  StringRef scope = "workgroup";

  Value base = LLVM::AddressOfOp::create(rewriter, loc, state);
  Value cntPtr =
      b.gep(ptrTy, i32_ty, base, ArrayRef<LLVM::GEPArg>{int32_t(barIdx)});

  Block *cur = rewriter.getInsertionBlock();
  Block *done = cur->splitBlock(rewriter.getInsertionPoint());
  Block *spin =
      rewriter.createBlock(cur->getParent(), std::next(Region::iterator(cur)));
  Block *arrive = rewriter.createBlock(
      cur->getParent(), std::next(Region::iterator(spin)));
  Block *poll = rewriter.createBlock(cur->getParent(),
                                     std::next(Region::iterator(arrive)));
  Block *skipPoll = rewriter.createBlock(
      cur->getParent(), std::next(Region::iterator(poll)));
  Block *mergePoll = rewriter.createBlock(
      cur->getParent(), std::next(Region::iterator(skipPoll)));
  mergePoll->addArgument(i1_ty, loc);
  Block *genericLast = nullptr;
  Block *genericObserve = nullptr;
  if (!useFixedWarpCount) {
    genericLast = rewriter.createBlock(
        cur->getParent(), std::next(Region::iterator(mergePoll)));
    genericObserve = rewriter.createBlock(
        cur->getParent(), std::next(Region::iterator(genericLast)));
  }

  rewriter.setInsertionPointToEnd(cur);
  // The BI lane register is already the warp-local lane ID.  Reading it
  // directly avoids materializing the CTA thread ID and masking it for every
  // software WS barrier.
  Value lane = LLVM::createLLVMIntrinsicCallOp(
                   rewriter, loc, "llvm.bi.lane.id", i32_ty, {})
                   .getResult(0);
  Value isLane0 = b.icmp_eq(lane, b.i32_val(0));
  LLVM::BrOp::create(rewriter, loc, spin);

  rewriter.setInsertionPointToEnd(spin);
  LLVM::CondBrOp::create(rewriter, loc, isLane0, arrive, skipPoll);

  // Arrival is a one-shot operation.  The counter value returned by this
  // atomic determines the phase target; retrying the atomic while polling
  // would count one leader multiple times and let a single warp advance the
  // phase on behalf of absent participants.
  rewriter.setInsertionPointToEnd(arrive);
  Value n = b.i32_val(static_cast<int32_t>(numWarps));
  // A fixed slot is a sequence of phases, each with exactly `numWarps`
  // warp-leader arrivals. Keep one unit ticket per arrival and round the
  // returned ticket to the current phase boundary. Since warp-specialize
  // partition sizes are powers of two, the fixed path needs only an AND plus
  // an add.
  Value ticket = LLVM::AtomicRMWOp::create(
                     rewriter, loc, LLVM::AtomicBinOp::add, cntPtr,
                     b.i32_val(1), LLVM::AtomicOrdering::release, scope)
                     .getResult();
  Value target;
  if (useFixedWarpCount) {
    assert(llvm::isPowerOf2_32(numWarps) &&
           "fixed warp-specialize participant count must be a power of two");
    Value phaseBase = b.and_(
        ticket, b.i32_val(-static_cast<int32_t>(numWarps)));
    target = b.add(phaseBase, n);
  } else {
    // Mixed slot reuse cannot round `ticket` by the current N: the previous
    // phase may have left the counter at a non-N-aligned value. Pack an
    // arrival count in the low byte and a monotonically increasing generation
    // above it. The last arrival resets the count and advances the generation;
    // all leaders wait for that generation transition.
    constexpr int32_t kArrivalMask = 0xff;
    constexpr int32_t kGenerationStep = 1 << 8;
    assert(numWarps < kGenerationStep &&
           "mixed warp-specialize participant count exceeds packed counter");
    Value phaseBase = b.and_(ticket, b.i32_val(~kArrivalMask));
    Value arrival = b.and_(ticket, b.i32_val(kArrivalMask));
    Value nextGeneration =
        b.add(phaseBase, b.i32_val(kGenerationStep));
    Value isLast = b.icmp_eq(
        arrival, b.i32_val(static_cast<int32_t>(numWarps - 1)));
    LLVM::CondBrOp::create(rewriter, loc, isLast, genericLast,
                           genericObserve);

    rewriter.setInsertionPointToEnd(genericLast);
    Value expected = b.add(ticket, b.i32_val(1));
    LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, cntPtr, expected, nextGeneration,
        LLVM::AtomicOrdering::release, LLVM::AtomicOrdering::monotonic, scope);
    LLVM::BrOp::create(rewriter, loc, genericObserve);

    target = nextGeneration;
    rewriter.setInsertionPointToEnd(genericObserve);
  }

  LLVM::BrOp::create(rewriter, loc, poll);

  rewriter.setInsertionPointToEnd(poll);
  Value curVal = LLVM::LoadOp::create(
      rewriter, loc, i32_ty, cntPtr, /*alignment=*/4,
      /*isVolatile=*/false, /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false, LLVM::AtomicOrdering::acquire, scope);
  Value reached = b.icmp_uge(curVal, target);
  LLVM::CondBrOp::create(rewriter, loc, reached, mergePoll,
                         ValueRange{b.i1_val(1)}, poll, ValueRange{});

  rewriter.setInsertionPointToEnd(skipPoll);
  LLVM::BrOp::create(rewriter, loc, ValueRange{b.i1_val(0)}, mergePoll);

  rewriter.setInsertionPointToStart(mergePoll);
  Value leaderDone = mergePoll->getArgument(0);
  Value allDone;
  if (threadsPerWarp == 64) {
    // The polling leader contributes true only after its acquire completes;
    // all other lanes contribute false at this warp reconvergence point.
    Value vote = LLVM::createLLVMIntrinsicCallOp(
                     rewriter, loc, "llvm.bi.vote.any", i32_ty,
                     {b.zext(i32_ty, leaderDone)})
                     .getResult(0);
    allDone = b.icmp_ne(vote, b.i32_val(0));
  } else {
    allDone = ::mlir::LLVM::ILUVATAR::shuffleIdx(
        loc, rewriter, leaderDone, 0);
  }
  LLVM::CondBrOp::create(rewriter, loc, allDone, done, spin);

  rewriter.setInsertionPointToStart(done);
}

static void createBarrier(TritonLLVMIRRewriter &b, LLVM::GlobalOp state,
                          unsigned threadsPerWarp, unsigned barIdx,
                          unsigned numThreads, bool useFixedWarpCount) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  if (numThreads <= threadsPerWarp)
    return;
  createNamedBarrier(b, b.getLoc(), state, threadsPerWarp, barIdx, numThreads,
                     useFixedWarpCount);
}

static void createAllBarrier(TritonLLVMIRRewriter &b, unsigned /*barIdx*/) {
  // `nvvm.barrier0` declares no memory effects, so MLIR treats it as pure and
  // CSEs adjacent occurrences. The warp-specialize dispatch protocol relies on
  // COUNTING these rendezvous: entering a region, the default group emits two
  // (release the workers, then wait until they have read the captures) and
  // each worker also passes two (switch-loop header plus partition entry). On
  // ivcore11 `createRegRealloc` is a no-op, so the default group's pair sat
  // adjacent, was merged into ONE barrier, and the two groups desynchronized
  // by one rendezvous: workers parked forever on their partition-entry
  // barrier and never ran their body, so a pipe consumer in the default
  // region spun on a commit that could not happen
  // (perf-iteration/ITERATION.md Trial 85).
  //
  // Give every dispatch rendezvous a distinct discardable attribute so CSE
  // cannot fold two of them together. This keeps the barrier count exact
  // without introducing extra memory fences, which the pipe's own fence
  // accounting asserts on.
  // The tag is derived from the insertion point's position in its block, so
  // it is deterministic across compilations (a process-global counter would
  // make the generated IR depend on compilation order).
  Block *block = b.getInsertionBlock();
  int32_t tag = static_cast<int32_t>(
      std::distance(block->begin(), b.getInsertionPoint()));
  auto barrier = NVVM::Barrier0Op::create(b, b.getLoc());
  barrier->setAttr("ws_rendezvous", b.getI32IntegerAttr(tag));
}

// A noinline Triton helper is lowered to a sibling LLVM function.  Its CTA
// barriers are therefore outside the syntactic ttg.warp_specialize region,
// even though only one warp group calls it.  Leaving such a barrier as a
// whole-CTA barrier deadlocks the other warp group.  Track only helpers that
// can reach a barrier, then clone that call graph for the execution scope of
// each warp group.
struct WarpGroupBarrierScope {
  unsigned barrierIdx;
  unsigned numThreads;
};

static llvm::DenseSet<Operation *>
collectBarrierDependentFunctions(ModuleOp module) {
  llvm::DenseMap<Operation *, SmallVector<Operation *>> callers;
  llvm::DenseSet<Operation *> dependent;
  SmallVector<Operation *> worklist;
  for (auto function : module.getOps<LLVM::LLVMFuncOp>()) {
    function.walk([&](NVVM::Barrier0Op) {
      if (dependent.insert(function).second)
        worklist.push_back(function);
    });
    function.walk([&](LLVM::CallOp call) {
      if (auto name = call.getCallee())
        if (auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*name))
          callers[callee].push_back(function);
    });
  }
  for (unsigned i = 0; i < worklist.size(); ++i)
    for (Operation *caller : callers[worklist[i]])
      if (dependent.insert(caller).second)
        worklist.push_back(caller);
  return dependent;
}

static std::string getScopedHelperName(LLVM::LLVMFuncOp callee,
                                       WarpGroupBarrierScope scope) {
  return (callee.getSymName() + "__tle_ws_barrier_" + Twine(scope.barrierIdx) +
          "_" + Twine(scope.numThreads))
      .str();
}

static void scopeWarpGroupHelperBarriers(
    ModuleOp module,
    ArrayRef<std::pair<LLVM::CallOp, WarpGroupBarrierScope>> rootCalls,
    const llvm::DenseSet<Operation *> &barrierDependent, LLVM::GlobalOp state,
    unsigned threadsPerWarp, const BarrierWarpCounts &barrierWarpCounts) {
  struct WorkItem {
    LLVM::LLVMFuncOp func;
    WarpGroupBarrierScope scope;
  };
  SmallVector<WorkItem> worklist;
  llvm::DenseSet<Operation *> visited;

  auto retargetCall = [&](LLVM::CallOp call, WarpGroupBarrierScope scope) {
    std::optional<StringRef> calleeName = call.getCallee();
    if (!calleeName)
      return;
    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*calleeName);
    if (!callee || callee.isExternal() || !barrierDependent.contains(callee))
      return;

    std::string scopedName = getScopedHelperName(callee, scope);
    auto scoped = module.lookupSymbol<LLVM::LLVMFuncOp>(scopedName);
    if (!scoped) {
      scoped = cast<LLVM::LLVMFuncOp>(callee.clone());
      scoped.setSymName(scopedName);
      OpBuilder builder(callee);
      builder.setInsertionPointAfter(callee);
      builder.insert(scoped);
    }
    call.setCalleeAttr(FlatSymbolRefAttr::get(module.getContext(), scopedName));
    if (visited.insert(scoped).second)
      worklist.push_back({scoped, scope});
  };

  for (auto [call, scope] : rootCalls)
    retargetCall(call, scope);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    LLVM::LLVMFuncOp scoped = worklist[i].func;
    WarpGroupBarrierScope scope = worklist[i].scope;

    SmallVector<LLVM::CallOp> nestedCalls;
    scoped.walk([&](LLVM::CallOp call) { nestedCalls.push_back(call); });
    for (LLVM::CallOp call : nestedCalls)
      retargetCall(call, scope);

    SmallVector<NVVM::Barrier0Op> barriers;
    scoped.walk([&](NVVM::Barrier0Op barrier) { barriers.push_back(barrier); });
    for (NVVM::Barrier0Op barrier : barriers) {
      TritonLLVMIRRewriter b(barrier.getLoc(), barrier);
      createBarrier(b, state, threadsPerWarp, scope.barrierIdx,
                    scope.numThreads,
                    barrierWarpCounts.lookup(scope.barrierIdx) != 0);
      barrier.erase();
    }
  }
}

// [WA] ivcore11 has no `setmaxnreg`, so register reallocation is a no-op.
static void createRegRealloc(TritonLLVMIRRewriter &, int, int) {}

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

#ifdef __ILUVATAR_TLE__
static bool isCtaInvariantSpecialRegister(Operation *op) {
  return isa<NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp,
             NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp,
             NVVM::ClusterIdXOp, NVVM::ClusterIdYOp, NVVM::ClusterIdZOp,
             NVVM::ClusterDimXOp, NVVM::ClusterDimYOp, NVVM::ClusterDimZOp,
             NVVM::BlockInClusterIdXOp, NVVM::BlockInClusterIdYOp,
             NVVM::BlockInClusterIdZOp>(op);
}
#endif

static LogicalResult findTrivialSubcomputation(LLVM::LLVMFuncOp func,
                                               Value capture,
                                               SetVector<Operation *> &ops) {
  SetVector<Value> worklist;
  worklist.insert(capture);
  for (unsigned i = 0; i != worklist.size(); ++i) {
    Value capture = worklist[i];
    // Check for a kernel argument.
    if (auto arg = dyn_cast<BlockArgument>(capture)) {
      if (arg.getOwner() == &func.getBody().front())
        continue;
      // Otherwise, this is some other block argument that cannot be elided.
      return failure();
    }

    Operation *op = capture.getDefiningOp();
#ifdef __ILUVATAR_TLE__
    // Special-register reads such as ctaid/nctaid are CTA-invariant values.
    // If they were explicitly captured by a warp-specialize op, preserve that
    // capture instead of rematerializing the read and its index arithmetic into
    // every partition.
    if (isCtaInvariantSpecialRegister(op))
      return failure();
#endif
    // Check if the defining op can be rematerialized. At the LLVM level,
    // checking for pure is probably a good enough heuristic.
    if (isPure(op)) {
      ops.insert(op);
      worklist.insert(op->operand_begin(), op->operand_end());
      continue;
    }
    // The op cannot be rematerialized.
    return failure();
  }

  // Cap the number of ops that can be rematerialized.
  // FIXME: This is arbitrary.
  return success(ops.size() <= 16);
}

static void elideTrivialCaptures(LLVM::LLVMFuncOp func,
                                 ArrayRef<WarpSpecializeOp> wsOps) {
  // The goal is to completely eliminate captures by hoisting or rematerializing
  // computations. We could minimize captures by rematerializing
  // subcomputations, but that is much more complicated. Prefer rematerializing
  // because that reduces liveranges. If subgraphs are duplicated more than
  // once, we will rely on CSE to clean them up.
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    llvm::BitVector toErase(wsOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(findTrivialSubcomputation(func, capture, subgraph)))
        continue;
      toErase.set(i);
      subgraph = topologicalSort(subgraph);

      for (Region *region : wsOp.getPartitionRegions()) {
        OpBuilder b(region);
        IRMapping mapping;
        for (Operation *op : subgraph) {
          b.clone(*op, mapping);
        }
        // Look the rematerialized value up through the capture's OWN defining
        // operation. Using `subgraph.back()` assumed the topological sort
        // ends on the op that produces `capture`, which does not hold when
        // the subgraph contains several mutually independent roots (e.g. a
        // pipe's separate full/empty `local_alloc` barrier arrays): the
        // captures were then cross-wired between partitions, so a reader
        // waited on the empty barrier while the writer committed to the
        // full one, deadlocking every pipe
        // (perf-iteration/ITERATION.md Trial 82).
        Value remat = capture;
        if (!subgraph.empty()) {
          auto captureResult = cast<OpResult>(capture);
          Operation *clonedDef = mapping.lookup(captureResult.getOwner());
          assert(clonedDef && "capture's defining op must have been cloned");
          remat = clonedDef->getResult(captureResult.getResultNumber());
        }
        region->getArgument(i).replaceAllUsesWith(remat);
      }
    }

    wsOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions()) {
      region->front().eraseArguments(toErase);
    }
  }
}

#ifdef __ILUVATAR_TLE__
static bool isHoistableCtaUniformLeaf(Operation *op) {
  return isCtaInvariantSpecialRegister(op) || isa<LLVM::ConstantOp>(op);
}

static LogicalResult findCtaUniformSubcomputation(LLVM::LLVMFuncOp func,
                                                  Value capture,
                                                  SetVector<Operation *> &ops) {
  SetVector<Value> worklist;
  worklist.insert(capture);
  for (unsigned i = 0; i != worklist.size(); ++i) {
    Value capture = worklist[i];
    if (auto arg = dyn_cast<BlockArgument>(capture)) {
      if (arg.getOwner() == &func.getBody().front())
        continue;
      return failure();
    }

    Operation *op = capture.getDefiningOp();
    if (!op)
      return failure();
    if (!op->getBlock() || op->getParentOfType<LLVM::LLVMFuncOp>() != func)
      return failure();

    // Only CTA-uniform special-register leaves may be hoisted into the common
    // warp-specialize header. Thread/lane/warp id reads are pure too, but they
    // are not CTA-uniform and must not be turned into shared partition values.
    if (op->getNumOperands() == 0 && !isHoistableCtaUniformLeaf(op))
      return failure();

    if (!isCtaInvariantSpecialRegister(op) && !isPure(op))
      return failure();

    ops.insert(op);
    worklist.insert(op->operand_begin(), op->operand_end());
  }

  return success(ops.size() <= 16);
}

static void hoistCtaUniformCapturesToHeader(LLVM::LLVMFuncOp func,
                                            ArrayRef<WarpSpecializeOp> wsOps,
                                            Block *header) {
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    llvm::BitVector toErase(wsOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(findCtaUniformSubcomputation(func, capture, subgraph)))
        continue;
      toErase.set(i);
      subgraph = topologicalSort(subgraph);

      Operation *terminator = header->getTerminator();
      for (Operation *op : subgraph) {
        if (op->getBlock() == header)
          continue;
        op->moveBefore(terminator);
      }

      for (Region *region : wsOp.getPartitionRegions())
        region->getArgument(i).replaceAllUsesWith(capture);
    }

    wsOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions())
      region->front().eraseArguments(toErase);
  }
}
#endif

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                              ArrayRef<WarpSpecializeOp> wsOps,
                                              unsigned threadsPerWarp,
                                              unsigned defaultWarpGroupSize,
                                              LLVM::GlobalOp state) {
  ModuleOp module = cast<ModuleOp>(func->getParentOp());
  auto barrierDependent = collectBarrierDependentFunctions(module);
  SmallVector<std::pair<LLVM::CallOp, WarpGroupBarrierScope>> helperCalls;

  SmallVector<NVVM::Barrier0Op> defaultBars;
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto call = dyn_cast<LLVM::CallOp>(op))
      helperCalls.push_back(
          {call, {kDefaultWarpGroupBarrierIdx, defaultWarpGroupSize}});

    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      if (!bar->hasAttr("sme_async_payload"))
        defaultBars.push_back(bar);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  BarrierWarpCounts barrierWarpCounts;
  for (NVVM::Barrier0Op bar : defaultBars)
    recordBarrierWarpCount(barrierWarpCounts, kSwitchLoopBarrierIdx,
                           defaultWarpGroupSize, threadsPerWarp);

  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      recordBarrierWarpCount(
          barrierWarpCounts, barIdx,
          threadsPerWarp * op.getPartitionNumWarps()[idx], threadsPerWarp);
    }
  }

  // Helper barriers are cloned later, but their execution scope is already
  // known from the call site.  Record all possible scopes before emitting any
  // barrier so a later reuse with a different warp count conservatively keeps
  // the unit-ticket protocol for that slot.
  for (auto [call, scope] : helperCalls) {
    auto calleeName = call.getCallee();
    if (!calleeName)
      continue;
    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*calleeName);
    if (callee && barrierDependent.contains(callee))
      recordBarrierWarpCount(barrierWarpCounts, scope.barrierIdx,
                             scope.numThreads, threadsPerWarp);
  }

  for (NVVM::Barrier0Op bar : defaultBars) {
    TritonLLVMIRRewriter b(bar.getLoc(), bar);
    // Slot 0 is used by helper-local barriers in the default group.  The
    // switch-loop barrier is a CTA barrier and does not consume this software
    // counter slot, so use slot 1 for inline default-region barriers.  Keeping
    // these static barrier streams separate prevents a layout-conversion
    // barrier from inheriting a generation from a preceding helper barrier.
    createBarrier(b, state, threadsPerWarp, kSwitchLoopBarrierIdx,
                  defaultWarpGroupSize,
                  barrierWarpCounts.lookup(kSwitchLoopBarrierIdx) != 0);
    bar.erase();
  }

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      unsigned warpGroupSize = threadsPerWarp * op.getPartitionNumWarps()[idx];
      SmallVector<NVVM::Barrier0Op> bars;
      partition->walk([&](LLVM::CallOp call) {
        helperCalls.push_back({call, {barIdx, warpGroupSize}});
      });
      partition->walk([&](NVVM::Barrier0Op bar) {
        // SME global-to-shared publication uses a real CTA rendezvous.  It
        // must not be rewritten to the partition-local software barrier,
        // because the producer and consumer are different warp groups.
        if (!bar->hasAttr("sme_async_payload"))
          bars.push_back(bar);
      });
      for (NVVM::Barrier0Op bar : bars) {
        TritonLLVMIRRewriter b(bar.getLoc(), bar);
        createBarrier(b, state, threadsPerWarp, barIdx, warpGroupSize,
                      barrierWarpCounts.lookup(barIdx) != 0);
        bar.erase();
      }
    }
  }

  scopeWarpGroupHelperBarriers(module, helperCalls, barrierDependent, state,
                               threadsPerWarp, barrierWarpCounts);

  return success();
}

static void rewritePartitionRegions(WarpSpecializeOp ws, Block *switchLoop,
                                    const TargetInfo &targetInfo, int lowRegs) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());

  for (Region *partition : ws.getPartitionRegions()) {
    // Load the explicit captures from shared memory and replace the block args
    // if there are any.
    b.setInsertionPointToStart(&partition->front());

    if (auto actRegs = ws.getActualRegisters()) {
      createRegRealloc(b, lowRegs,
                       (*actRegs)[partition->getRegionNumber() + 1]);
    }

    if (partition->getNumArguments()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partition->getArgumentTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
      for (auto [i, arg] :
           llvm::zip(llvm::seq<int32_t>(partition->getNumArguments()),
                     partition->getArguments())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        // Each thread in the warp group needs a copy of the value.
        Value value = b.load(arg.getType(), ptr, /*align=*/1);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The shared memory is only live for the entry into the region, so put
    // another barrier here.
    createAllBarrier(b, kSwitchLoopBarrierIdx);

    // Rewrite all warp returns.
    partition->walk([&](WarpReturnOp op) {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      createAllBarrier(b, kSwitchLoopBarrierIdx);
      if (auto actRegs = ws.getActualRegisters()) {
        createRegRealloc(b, (*actRegs)[partition->getRegionNumber() + 1],
                         lowRegs);
      }
      b.replaceOpWithNewOp<LLVM::BrOp>(op, switchLoop);
    });
  }
}

// A CTA barrier cannot be placed at different program counters in two
// divergent WS regions on ivcore11. The hardware accepts the instruction
// there, but SME writes are not reliably published. Keep each region's CFG
// intact and call one shared noinline/convergent helper instead; this gives the
// barrier one program counter without invalidating loop-carried SSA values.
static void annotateSmeBarrierMarkers(ArrayRef<WarpSpecializeOp> wsOps) {
  Builder builder(wsOps.front()->getContext());
  for (auto [wsId, wsOp] : llvm::enumerate(wsOps)) {
    WarpSpecializeOp ws = wsOp;
    auto startIds = ws.getWarpGroupStartIds();
    SmallVector<unsigned> partitionGroups(ws.getPartitionRegions().size());
    SmallVector<unsigned> groupStarts;
    if (startIds && startIds->size() == partitionGroups.size()) {
      SmallVector<unsigned> order(partitionGroups.size());
      std::iota(order.begin(), order.end(), 0);
      llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
        return (*startIds)[lhs] < (*startIds)[rhs];
      });
      unsigned group = 0;
      for (unsigned index : order) {
        unsigned start = (*startIds)[index];
        if (groupStarts.empty() || start % 4 == 0) {
          groupStarts.push_back(start);
          ++group;
        }
        partitionGroups[index] = group;
      }
    } else {
      groupStarts.resize(partitionGroups.size());
      for (auto [index, start] : llvm::enumerate(groupStarts)) {
        partitionGroups[index] = index + 1;
        start = 0;
      }
    }
    unsigned groupCount = groupStarts.size() + 1;
    auto mark = [&](NVVM::Barrier0Op bar, unsigned group, unsigned startWarp,
                    unsigned numWarps) {
      bar->setAttr("sme_ws_id", builder.getI32IntegerAttr(wsId));
      bar->setAttr("sme_ws_group", builder.getI32IntegerAttr(group));
      bar->setAttr("sme_ws_group_count",
                   builder.getI32IntegerAttr(groupCount));
      bar->setAttr("sme_ws_group_start_warp",
                   builder.getI32IntegerAttr(startWarp));
      bar->setAttr("sme_ws_group_warps",
                   builder.getI32IntegerAttr(numWarps));
    };
    ws.getDefaultRegion().walk([&](NVVM::Barrier0Op bar) {
      if (bar->hasAttr("sme_async_payload"))
        mark(bar, 0, 0, lookupNumWarps(ws));
    });
    for (auto [idx, partition] : llvm::enumerate(ws.getPartitionRegions()))
      partition->walk([&](NVVM::Barrier0Op bar) {
        if (bar->hasAttr("sme_async_payload"))
          mark(bar, partitionGroups[idx], groupStarts[partitionGroups[idx] - 1],
               ws.getPartitionNumWarps()[idx]);
      });
  }
}

// Introducing a common rendezvous changes dominance even when the warp-id
// dispatch preserves each group's dynamic path. Rebuild the affected SSA uses
// with block arguments, including loop headers reached through new backedges.
static LogicalResult repairSmeRendezvousSSA(LLVM::LLVMFuncOp func) {
  struct Use {
    Operation *owner;
    unsigned index;
    std::optional<unsigned> successor;
  };
  auto getUseOperand = [](const Use &use) -> OpOperand & {
    if (!use.successor)
      return use.owner->getOpOperand(use.index);
    auto branch = cast<BranchOpInterface>(use.owner);
    return branch.getSuccessorOperands(*use.successor)
        .getMutableForwardedOperands()[use.index];
  };
  DominanceInfo dominance(func);
  DenseMap<Value, SmallVector<Use>> invalidUses;
  func.walk([&](Operation *op) {
    auto branch = dyn_cast<BranchOpInterface>(op);
    for (OpOperand &operand : op->getOpOperands()) {
      if (dominance.dominates(operand.get(), op))
        continue;
      unsigned operandNumber = operand.getOperandNumber();
      std::optional<unsigned> successor;
      unsigned successorIndex = 0;
      if (branch) {
        for (unsigned i = 0, e = branch->getNumSuccessors(); i < e; ++i) {
          auto forwarded =
              branch.getSuccessorOperands(i).getForwardedOperands();
          if (forwarded.empty())
            continue;
          unsigned begin = forwarded.getBeginOperandIndex();
          if (operandNumber >= begin &&
              operandNumber < begin + forwarded.size()) {
            successor = i;
            successorIndex = operandNumber - begin;
            break;
          }
        }
      }
      invalidUses[operand.get()].push_back(
          {op, successor ? successorIndex : operandNumber, successor});
    }
  });

  for (auto &[value, uses] : invalidUses) {
    if (auto constant = value.getDefiningOp<LLVM::ConstantOp>()) {
      // LLVM target intrinsics such as Iluvatar STP encode selected operands
      // as immarg. Rematerialize constants at each use instead of threading
      // them through a CFG phi, which would make the translation invalid.
      for (const Use &record : uses) {
        OpOperand &use = getUseOperand(record);
        OpBuilder builder(record.owner);
        builder.setInsertionPoint(record.owner);
        Value replacement = LLVM::ConstantOp::create(
            builder, value.getLoc(), value.getType(), constant.getValueAttr());
        use.set(replacement);
      }
      continue;
    }
    Block *defBlock = value.getParentBlock();
    DenseMap<Block *, Value> available;
    available[defBlock] = value;
    std::function<FailureOr<Value>(Block *)> getAvailable =
        [&](Block *block) -> FailureOr<Value> {
      if (auto it = available.find(block); it != available.end())
        return it->second;
      if (block->hasNoPredecessors()) {
        // The warp-id dispatch makes this edge unreachable for the value's
        // warp group, but MLIR's scalar CFG does not encode that predicate.
        // Keep the impossible edge well-formed without changing the selected
        // group's value flow; this is the same SSA convention used for a
        // predicated LLVM branch whose inactive arm is never consumed.
        OpBuilder builder = OpBuilder::atBlockBegin(block);
        Value undef = LLVM::UndefOp::create(builder, value.getLoc(),
                                           value.getType());
        available[block] = undef;
        return undef;
      }

      // Cache the block argument before visiting predecessors so cyclic CFGs
      // terminate and transport loop-carried values through the rendezvous.
      Value arg = block->addArgument(value.getType(), value.getLoc());
      available[block] = arg;
      SmallVector<Block *> predecessors(block->getPredecessors());
      llvm::sort(predecessors);
      predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                         predecessors.end());
      for (Block *pred : predecessors) {
        auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
        if (!branch) {
          func.emitError("cannot repair SSA across SME rendezvous: incoming "
                         "block has no branch terminator")
              << "value type=" << value.getType()
              << ", terminator=" << pred->getTerminator()->getName();
          return failure();
        }
        FailureOr<Value> incoming = getAvailable(pred);
        if (failed(incoming)) {
          func.emitError("cannot repair SSA across SME rendezvous: value "
                         "has no incoming path")
              << "type=" << value.getType();
          return failure();
        }
        for (auto [index, successor] : llvm::enumerate(branch->getSuccessors()))
          if (successor == block)
            branch.getSuccessorOperands(index).append(ValueRange{*incoming});
      }
      return arg;
    };

    for (const Use &record : uses) {
      OpOperand &use = getUseOperand(record);
      FailureOr<Value> replacement = getAvailable(record.owner->getBlock());
      if (failed(replacement))
        return func.emitError("cannot repair SSA across SME rendezvous");
      use.set(*replacement);
    }
  }
  return success();
}

static LogicalResult convergeSmeBarrierSet(
    LLVM::LLVMFuncOp func,
    ArrayRef<std::pair<unsigned, NVVM::Barrier0Op>> markers,
    unsigned threadsPerWarp, const TargetInfo &targetInfo) {
  if (markers.size() < 2)
    return func.emitError("SME publication barrier needs at least two WS "
                          "warp groups");

  Location loc = markers.front().second->getLoc();
  SmallVector<Block *> continuations;
  SmallVector<unsigned> groupStarts;
  Block *join = new Block;
  func.getBody().push_back(join);

  for (auto [groupId, marker] : markers) {
    auto startAttr =
        marker->getAttrOfType<IntegerAttr>("sme_ws_group_start_warp");
    if (!startAttr)
      return func.emitError("SME publication marker lacks warp-group geometry");
    groupStarts.push_back(static_cast<unsigned>(startAttr.getInt()));
    Block *block = marker->getBlock();
    continuations.push_back(block->splitBlock(std::next(marker->getIterator())));
    marker.erase();
    OpBuilder predecessor(block, block->end());
    LLVM::BrOp::create(predecessor, loc, join);
  }

  // All groups execute the same full CTA barrier before the warp-uniform
  // dispatch resumes their original continuation.
  TritonLLVMIRRewriter joinBuilder(loc, OpBuilder::atBlockBegin(join));
  NVVM::Barrier0Op::create(joinBuilder, loc);
  Value tid = NVVM::ThreadIdXOp::create(
      joinBuilder, loc, joinBuilder.getIntegerType(32));
  Value wid = joinBuilder.udiv(tid, joinBuilder.i32_val(threadsPerWarp));
  // Match the warp-uniform value used by the normal WS dispatch header.  The
  // join is reached from multiple divergent groups, so the raw thread-id
  // quotient must be normalized before it controls the continuation branch.
  wid = targetInfo.shuffleIdx(joinBuilder, loc, wid, 0);
  Block *dispatch = join;
  for (unsigned group = 0; group + 1 < continuations.size(); ++group) {
    Block *next = new Block;
    func.getBody().push_back(next);
    TritonLLVMIRRewriter dispatchBuilder(
        loc, OpBuilder::atBlockEnd(dispatch));
    Value inGroup = dispatchBuilder.icmp_ult(
        wid, dispatchBuilder.i32_val(groupStarts[group + 1]));
    LLVM::CondBrOp::create(dispatchBuilder, loc, inGroup,
                           continuations[group], next);
    dispatch = next;
  }
  TritonLLVMIRRewriter tailBuilder(loc, OpBuilder::atBlockEnd(dispatch));
  LLVM::BrOp::create(tailBuilder, loc, continuations.back());
  return success();
}

static LogicalResult rewriteSmePublicationBarriers(LLVM::LLVMFuncOp func) {
  using Marker = std::pair<unsigned, NVVM::Barrier0Op>;
  DenseMap<unsigned, DenseMap<unsigned, SmallVector<NVVM::Barrier0Op>>>
      grouped;
  DenseMap<unsigned, unsigned> expectedGroups;
  func.walk([&](NVVM::Barrier0Op bar) {
    if (!bar->hasAttr("sme_async_payload"))
      return;
    auto wsId = bar->getAttrOfType<IntegerAttr>("sme_ws_id");
    auto group = bar->getAttrOfType<IntegerAttr>("sme_ws_group");
    auto count = bar->getAttrOfType<IntegerAttr>("sme_ws_group_count");
    if (!wsId || !group || !count)
      return;
    grouped[wsId.getInt()][group.getInt()].push_back(bar);
    expectedGroups[wsId.getInt()] = count.getInt();
  });

  ModuleOp module = func->getParentOfType<ModuleOp>();
  for (auto &[wsId, groups] : grouped) {
    unsigned groupCount = expectedGroups.lookup(wsId);
    for (unsigned group = 0; group < groupCount; ++group) {
      if (!groups.count(group))
        return func.emitError()
               << "SME publication barrier in WS " << wsId
               << " is missing warp group " << group;
    }

    unsigned markerCount = groups.begin()->second.size();
    for (auto &[group, markers] : groups)
      if (markers.size() != markerCount) {
        return func.emitError()
               << "SME publication barriers in WS " << wsId
               << " have inconsistent per-group counts: group " << group
               << " has " << markers.size() << ", expected " << markerCount;
      }

    for (unsigned index = 0; index < markerCount; ++index)
      for (unsigned group = 0; group < groupCount; ++group) {
        NVVM::Barrier0Op marker = groups[group][index];
        OpBuilder builder(marker);
        // Keep the ALU publication primitive at the original WS-region
        // program point.  A noinline helper has one call target but two
        // divergent callers (default and partition); on ivcore11 that makes
        // the convergent instruction participate in the wrong region-level
        // rendezvous.  Direct emission preserves the hardware instruction
        // while keeping its control-flow scope explicit.
        // Keep the ALU publication primitive at the original WS-region
        // program point.  A noinline helper has one call target but two
        // divergent callers (default and partition); on ivcore11 that makes
        // the convergent instruction participate in the wrong region-level
        // rendezvous.  Direct emission preserves the hardware instruction
        // while keeping its control-flow scope explicit.
        LLVM::createLLVMIntrinsicCallOp(builder, marker.getLoc(),
                                        "llvm.bi.sl.barrier.alu", {}, {});
        marker.erase();
      }
  }
  return success();
}

// LLVM's LICM will be tempted to hoist code out of the switch loop generated by
// the `ttg.warp_specialize` lowering. However, neither NVPTX or `ptxas` will
// rematerialize this code back in to the partition regions, resulting in long
// liveranges for an arbitrary number of registers.
//
// Due to reduced warp group registers, these live values can induce spilling
// in the partition regions. Prevent this by disabling LICM on the switch loop.
static void disableLICM(LLVM::BrOp latchBr) {
  Builder b(latchBr.getContext());
  MLIRContext *ctx = b.getContext();
  auto licmMD = LLVM::LoopLICMAttr::get(ctx, b.getBoolAttr(true), {});
  auto loopMD =
      LLVM::LoopAnnotationAttr::get(b.getContext(), {}, {}, {}, {}, {}, licmMD,
                                    {}, {}, {}, {}, {}, {}, {}, {}, {});
  latchBr.setLoopAnnotationAttr(loopMD);
}

static void initEmulatedNamedBarrierState(TritonLLVMIRRewriter &b, Value tid,
                                          LLVM::GlobalOp state,
                                          unsigned numBarriers) {
  MLIRContext *ctx = b.getContext();
  Block *cur = b.getInsertionBlock();
  Block *cont = cur->splitBlock(b.getInsertionPoint());
  Block *init =
      b.createBlock(cur->getParent(), std::next(Region::iterator(cur)));

  b.setInsertionPointToEnd(cur);
  Value isThread0 = b.icmp_eq(tid, b.i32_val(0));
  LLVM::CondBrOp::create(b, b.getLoc(), isThread0, init, cont);

  b.setInsertionPointToEnd(init);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
  Value base = LLVM::AddressOfOp::create(b, b.getLoc(), state);
  Value zero = b.i32_val(0);
  for (unsigned i = 0; i < numBarriers; ++i) {
    Value ptr =
        b.gep(ptrTy, b.getIntegerType(32), base,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(i)});
    b.store(zero, ptr);
  }
  LLVM::BrOp::create(b, b.getLoc(), cont);

  b.setInsertionPointToStart(cont);
  NVVM::Barrier0Op::create(b, b.getLoc());
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const TargetInfo &targetInfo,
                                         LLVM::GlobalOp state,
                                         unsigned numBarriers) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  annotateSmeBarrierMarkers(wsOps);

  // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  unsigned defaultNumWarps = lookupNumWarps(func);
  unsigned defaultWarpGroupSize = threadsPerWarp * defaultNumWarps;
  if (failed(rewriteWarpGroupBarriers(func, wsOps, threadsPerWarp,
                                      defaultWarpGroupSize, state)))
    return failure();

  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
  }

  // [WA] ivcore11 has no dynamic register reallocation; keep register
  // bookkeeping disabled so `createRegRealloc` stays a no-op.
  int lowRegs = -1;
  int defRegs = -1;

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
  elideTrivialCaptures(func, wsOps);

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = NVVM::ThreadIdXOp::create(b, b.getLoc(), i32_ty);

  initEmulatedNamedBarrierState(b, tid, state, numBarriers);

  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  // Tell the backend this value is warp-uniform.
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });
#ifdef __ILUVATAR_TLE__
  hoistCtaUniformCapturesToHeader(func, wsOps, header);
#endif

  // ^switchLoop:
  //   barrier (all)
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_wid = sub %wid, <default_warp_group>
  b.setInsertionPointToStart(switchLoop);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  Value relWid = b.sub(wid, b.i32_val(defaultNumWarps));

  // The default warp group populates the state pointer with the state ID for
  // all warps.
  LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);
  Value warpStatePtr = b.gep(ptrTy, i8_ty, statePtr, relWid);
  // All threads in a warp reading from the same smem address will not create
  // bank conflicts and is better than predicated load.
  Value warpState = b.load(i8_ty, warpStatePtr);

  // Pull the partition regions out. Switch based on the state ID to the right
  // partition.
  SmallVector<Block *> partitionBlocks;
  SmallVector<int32_t> partitionStates;
  int32_t partitionStateCounter = 0;
  // This represents the data that the default warp group will fill into the
  // state pointer before entering each `warp_specialize` region, which maps
  // a warp ID to a state ID in the switch.
  int32_t maxNumWarps = totalNumWarpsAttr.getInt() - defaultNumWarps;
  SmallVector<SmallVector<int32_t>> warpToState(
      wsOps.size(), SmallVector<int32_t>(maxNumWarps, -1));
  for (auto [op, stateMap] : llvm::zip(wsOps, warpToState)) {
    rewritePartitionRegions(op, switchLoop, targetInfo, lowRegs);
    for (auto [partition, partitionNumWarps, startId] :
         llvm::zip(op.getPartitionRegions(), op.getPartitionNumWarps(),
                   *op.getWarpGroupStartIds())) {
      partitionStates.push_back(partitionStateCounter++);
      partitionBlocks.push_back(&partition->front());
      for (int32_t &stateId : MutableArrayRef(stateMap).slice(
               startId - defaultNumWarps, partitionNumWarps))
        stateId = partitionStates.back();
    }
  }
  if (partitionStateCounter > std::numeric_limits<uint8_t>::max()) {
    return mlir::emitError(func.getLoc(),
                           "FIXME: too many warp group partitions");
  }

  // Splice them in reverse order so the IR is easier to read.
  Region::BlockListType &funcBlocks = func.getBody().getBlocks();
  for (Block *block : llvm::reverse(partitionBlocks)) {
    Region *region = block->getParent();
    funcBlocks.splice(std::next(switchLoop->getIterator()),
                      region->getBlocks());
  }

  // Default destination.
  Block *defaultBlock = new Block;
  funcBlocks.insert(std::next(switchLoop->getIterator()), defaultBlock);
  b.setInsertionPointToStart(defaultBlock);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  auto latchBr = LLVM::BrOp::create(b, b.getLoc(), switchLoop);
  disableLICM(latchBr);

  // Exit state.
  Block *switchExit = new Block;
  funcBlocks.insert(std::next(defaultBlock->getIterator()), switchExit);
  partitionBlocks.push_back(switchExit);
  partitionStates.push_back(partitionStateCounter);

  // Create the switch.
  b.setInsertionPointToEnd(switchLoop);
  SmallVector<APInt> caseValues;
  for (int32_t state : partitionStates)
    caseValues.push_back(APInt(8, state));
  LLVM::SwitchOp::create(b, b.getLoc(), warpState, defaultBlock, ValueRange(),
                         caseValues, partitionBlocks,
                         SmallVector<ValueRange>(partitionBlocks.size()));

  // Now add synchronization around the default regions.
  for (auto [ws, stateMap] : llvm::zip(wsOps, warpToState)) {
    Block *before = ws->getBlock();
    Block *after = b.splitBlock(before, ws->getIterator());
    TritonLLVMIRRewriter b(ws.getLoc(), OpBuilder::atBlockEnd(before));
    Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    for (auto [i, state] : llvm::enumerate(stateMap)) {
      Value stateVal = b.i8_val(state);
      b.store(stateVal, b.gep(ptrTy, i8_ty, statePtr, LLVM::GEPArg(i)));
    }

    // Store the captures if there are any.
    if (ws.getNumOperands()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(ws.getOperandTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      for (auto [i, arg] : llvm::zip(llvm::seq<int32_t>(ws.getNumOperands()),
                                     ws.getOperands())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        b.store(arg, ptr, /*align=*/1);
      }
    }

    // First barrier releases the waiting warpgroups. The second barrier ensures
    // they have read the captures before the memory is released upon entry.
    createAllBarrier(b, kSwitchLoopBarrierIdx);
    if (auto actRegs = ws.getActualRegisters())
      createRegRealloc(b, defRegs, actRegs->front());
    createAllBarrier(b, kSwitchLoopBarrierIdx);
    LLVM::BrOp::create(b, b.getLoc(), &ws.getDefaultRegion().front());

    ws.getDefaultRegion().walk([&, ws = ws](WarpYieldOp op) mutable {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      createAllBarrier(b, kSwitchLoopBarrierIdx);
      if (auto actRegs = ws.getActualRegisters())
        createRegRealloc(b, actRegs->front(), defRegs);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, op.getOperands(), after);
    });
    after->getParent()->getBlocks().splice(after->getIterator(),
                                           ws.getDefaultRegion().getBlocks());

    // Replace the results.
    auto outputs = after->addArguments(
        ws.getResultTypes(),
        SmallVector<Location>(ws.getNumResults(), ws.getLoc()));
    ws.replaceAllUsesWith(outputs);
    ws.erase();
  }

  if (failed(rewriteSmePublicationBarriers(func)))
    return failure();

  // Signal all warp groups to exit.
  func.walk([&](LLVM::ReturnOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    Value cst = b.i8_val(partitionStateCounter);
    for (int32_t i : llvm::seq(maxNumWarps))
      b.store(cst, b.gep(ptrTy, i8_ty, statePtr, LLVM::GEPArg(i)));
    createAllBarrier(b, kSwitchLoopBarrierIdx);
  });
  b.setInsertionPointToStart(switchExit);
  LLVM::ReturnOp::create(b, b.getLoc(), ValueRange());

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ILUVATARWarpSpecializeToLLVM
    : public mlir::triton::impl::ILUVATARWarpSpecializeToLLVMBase<
          ILUVATARWarpSpecializeToLLVM> {

  explicit ILUVATARWarpSpecializeToLLVM(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool hasWS = false;
    mod.walk([&](WarpSpecializeOp) {
      hasWS = true;
      return WalkResult::interrupt();
    });
    if (!hasWS)
      return;

    std::string archStr = this->arch.getValue();
    if (archStr != "ivcore11") {
      mlir::emitError(mod.getLoc()) << "ttg.warp_specialize lowering is only "
                                       "supported on ivcore11, arch '"
                                    << archStr << "' is not supported yet";
      return signalPassFailure();
    }

    mlir::emitRemark(mod.getLoc())
        << "[WA] lowering ttg.warp_specialize on ivcore11 with shared-memory "
           "emulate named barriers (no hardware named barrier / setmaxnreg)";

    TargetInfo targetInfo(archStr);

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(&getContext(), option,
                                               targetInfo);
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        convertOpTypes(op, typeConverter);
    });
    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();

    unsigned numBarriers = kNumReservedBarriers;
    mod.walk([&](WarpSpecializeOp op) {
      numBarriers = std::max<unsigned>(
          numBarriers, kNumReservedBarriers + op.getPartitionRegions().size());
    });
    LLVM::GlobalOp state = getOrCreateSwBarrierState(mod, numBarriers);

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo, state, numBarriers)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createILUVATARWarpSpecializeToLLVMPass(StringRef targetArch) {
  return std::make_unique<ILUVATARWarpSpecializeToLLVM>(targetArch);
}
