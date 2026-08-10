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
// NVIDIA's warp specialization relies on two hardware features that ivcore11
// does NOT provide:
//   1. Named / partial `barrier.sync <id>, <numThreads>` barriers, which let a
//      *subset* of the CTA (a single warp group) synchronize independently.
//   2. Dynamic per-warp register reallocation (`setmaxnreg`).
//
// The emulate named barrier trades performance for correctness; it is only
// meant as a functional bring-up path for the current architecture.
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "TritonILUVATARGPUToLLVM/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

static LLVM::GlobalOp getOrCreateSwBarrierState(ModuleOp mod) {
  if (auto g = dyn_cast_or_null<LLVM::GlobalOp>(
          mod.lookupSymbol(kNamedBarStateName)))
    return g;
  OpBuilder rewriter(mod.getBodyRegion());
  auto arrTy = LLVM::LLVMArrayType::get(i32_ty, kNumBarriers);
  return LLVM::GlobalOp::create(
      rewriter, mod.getLoc(), arrTy, /*isConstant=*/false,
      LLVM::Linkage::Internal, kNamedBarStateName, /*value=*/Attribute(),
      /*alignment=*/4, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
}

// [WA] Emulate a named barrier that synchronizes exactly `numThreads` threads
// sharing slot `barIdx`.
static void createEmulatedNamedBarrier(RewriterBase &rewriter, Location loc,
                                       LLVM::GlobalOp state,
                                       unsigned threadsPerWarp, unsigned barIdx,
                                       unsigned numThreads) {
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

  // CFG:  cur -isLane0-> ^arrive -> ^spin -done-> ^done
  //            \-else----------------------------/
  Block *cur = rewriter.getInsertionBlock();
  Block *done = cur->splitBlock(rewriter.getInsertionPoint());
  Block *arrive =
      rewriter.createBlock(cur->getParent(), std::next(Region::iterator(cur)));
  Block *spin = rewriter.createBlock(cur->getParent(),
                                     std::next(Region::iterator(arrive)));

  rewriter.setInsertionPointToEnd(cur);
  Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
  Value lane = b.and_(tid, b.i32_val(static_cast<int32_t>(threadsPerWarp - 1)));
  Value isLane0 = b.icmp_eq(lane, b.i32_val(0));
  LLVM::CondBrOp::create(rewriter, loc, isLane0, arrive, done);

  rewriter.setInsertionPointToEnd(arrive);
  Value ticket = LLVM::AtomicRMWOp::create(
                     rewriter, loc, LLVM::AtomicBinOp::add, cntPtr,
                     b.i32_val(1), LLVM::AtomicOrdering::release, scope)
                     .getResult();
  // target = ticket - (ticket % numWarps) + numWarps
  Value n = b.i32_val(static_cast<int32_t>(numWarps));
  Value round = b.udiv(ticket, n);
  Value target = b.mul(b.add(round, b.i32_val(1)), n);
  LLVM::BrOp::create(rewriter, loc, spin);

  rewriter.setInsertionPointToEnd(spin);
  Value curVal = LLVM::AtomicRMWOp::create(
                     rewriter, loc, LLVM::AtomicBinOp::add, cntPtr,
                     b.i32_val(0), LLVM::AtomicOrdering::acquire, scope)
                     .getResult();
  Value reached = b.icmp_uge(curVal, target);
  LLVM::CondBrOp::create(rewriter, loc, reached, done, spin);

  rewriter.setInsertionPointToStart(done);
  // LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.bi.sch.barrier",
  //                                 TypeRange(), ValueRange());
}

static void createBarrier(TritonLLVMIRRewriter &b, LLVM::GlobalOp state,
                          unsigned threadsPerWarp, unsigned barIdx,
                          unsigned numThreads) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  if (numThreads <= threadsPerWarp)
    return;
  createEmulatedNamedBarrier(b, b.getLoc(), state, threadsPerWarp, barIdx,
                             numThreads);
}

static void createAllBarrier(TritonLLVMIRRewriter &b, unsigned /*barIdx*/) {
  NVVM::Barrier0Op::create(b, b.getLoc());
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
        Value remat = capture;
        if (!subgraph.empty()) {
          unsigned resultIdx = cast<OpResult>(capture).getResultNumber();
          remat = mapping.lookup(subgraph.back())->getResult(resultIdx);
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
  SmallVector<NVVM::Barrier0Op> defaultBars;
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      defaultBars.push_back(bar);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  for (NVVM::Barrier0Op bar : defaultBars) {
    TritonLLVMIRRewriter b(bar.getLoc(), bar);
    createBarrier(b, state, threadsPerWarp, kDefaultWarpGroupBarrierIdx,
                  defaultWarpGroupSize);
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
      partition->walk([&](NVVM::Barrier0Op bar) { bars.push_back(bar); });
      for (NVVM::Barrier0Op bar : bars) {
        TritonLLVMIRRewriter b(bar.getLoc(), bar);
        createBarrier(b, state, threadsPerWarp, barIdx, warpGroupSize);
        bar.erase();
      }
    }
  }

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

// [WA] Zero the software-barrier counters at kernel entry (shared memory is
// uninitialized on the GPU). Thread 0 writes all slots, then the whole CTA
// synchronizes so every warp observes the zeroed state before it can reach any
// emulate named barrier.
static void initEmulatedNamedBarrierState(TritonLLVMIRRewriter &b, Value tid,
                                          LLVM::GlobalOp state) {
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
  for (int32_t i = 0; i < kNumBarriers; ++i) {
    Value ptr =
        b.gep(ptrTy, b.getIntegerType(32), base, ArrayRef<LLVM::GEPArg>{i});
    b.store(zero, ptr);
  }
  LLVM::BrOp::create(b, b.getLoc(), cont);

  b.setInsertionPointToStart(cont);
  NVVM::Barrier0Op::create(b, b.getLoc());
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const TargetInfo &targetInfo,
                                         LLVM::GlobalOp state) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

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

  // [WA] Initialize the shared emulated named barrier state before anyone can
  // use it.
  initEmulatedNamedBarrierState(b, tid, state);

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

    LLVM::GlobalOp state = getOrCreateSwBarrierState(mod);

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo, state)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createILUVATARWarpSpecializeToLLVMPass(StringRef targetArch) {
  return std::make_unique<ILUVATARWarpSpecializeToLLVM>(targetArch);
}
