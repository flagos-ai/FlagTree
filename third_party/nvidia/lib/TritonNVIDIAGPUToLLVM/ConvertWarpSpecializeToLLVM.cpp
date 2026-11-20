#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPSPECIALIZETOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

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

// Original FlagTree implementation before the Triton #9366-style backport
// (kept for review context):
//
// static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
//                           unsigned numThreads) {
//   assert(barIdx < kNumBarriers && "not enough barriers");
//   // If a partition has only 1 warp, use `bar.warp.sync`.
//   if (numThreads == 32)
//     LLVM::NVIDIA::createSyncWarp(b.getLoc(), b);
//   else
//     NVVM::BarrierOp::create(b, b.getLoc(), b.i32_val(barIdx),
//                             b.i32_val(numThreads));
// }
//
// Adapted from https://github.com/triton-lang/triton/pull/9366 (commit
// f1ece2c, "ws function calls"). Accepting the barrier id as an SSA value lets
// an outlined helper receive its caller's warp-group barrier as a hidden
// argument. The unsigned overload preserves existing constant-id call sites.
static void createBarrier(TritonLLVMIRRewriter &b, Value barIdx,
                          unsigned numThreads) {
  // If a partition has only 1 warp, use `bar.warp.sync`.
  if (numThreads == 32)
    LLVM::NVIDIA::createSyncWarp(b.getLoc(), b);
  else
    NVVM::BarrierOp::create(b, b.getLoc(), TypeRange{}, barIdx,
                            b.i32_val(numThreads), {}, Value{});
}

static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
                          unsigned numThreads) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  createBarrier(b, b.i32_val(barIdx), numThreads);
}

static void createAllBarrier(TritonLLVMIRRewriter &b, unsigned barIdx) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  LLVM::createLLVMIntrinsicCallOp(b, b.getLoc(),
                                  "llvm.nvvm.barrier.cta.sync.all",
                                  void_ty(b.getContext()), b.i32_val(barIdx));
}

//===----------------------------------------------------------------------===//
#ifdef __TLE__
// TLE variant of elideTrivialCaptures
//===----------------------------------------------------------------------===//

static bool isCtaInvariantSpecialRegister(Operation *op) {
  return isa<NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp,
             NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp,
             NVVM::ClusterIdXOp, NVVM::ClusterIdYOp, NVVM::ClusterIdZOp,
             NVVM::ClusterDimXOp, NVVM::ClusterDimYOp, NVVM::ClusterDimZOp,
             NVVM::BlockInClusterIdXOp, NVVM::BlockInClusterIdYOp,
             NVVM::BlockInClusterIdZOp>(op);
}

static LogicalResult tleFindTrivialSubcomputation(LLVM::LLVMFuncOp func,
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
    // Special-register reads such as ctaid/nctaid are CTA-invariant values.
    // If they were explicitly captured by a warp-specialize op, preserve that
    // capture instead of rematerializing the read and its index arithmetic into
    // every partition.
    if (isCtaInvariantSpecialRegister(op))
      return failure();
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

static void tleElideTrivialCaptures(LLVM::LLVMFuncOp func,
                                    ArrayRef<WarpSpecializeOp> wsOps) {
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    auto partOp = wsOp.getPartitionOp();
    llvm::BitVector toErase(partOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(partOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(tleFindTrivialSubcomputation(func, capture, subgraph)))
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

    partOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions()) {
      region->front().eraseArguments(toErase);
    }
  }
}

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
    auto partOp = wsOp.getPartitionOp();
    llvm::BitVector toErase(partOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(partOp.getExplicitCaptures())) {
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

    partOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions())
      region->front().eraseArguments(toErase);
  }
}

//===----------------------------------------------------------------------===//
#endif
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

static void createRegRealloc(TritonLLVMIRRewriter &b, int curRegs,
                             int adjRegs) {
  curRegs = std::min(256, curRegs);
  adjRegs = std::min(256, adjRegs);
  auto action = adjRegs < curRegs ? NVVM::SetMaxRegisterAction::decrease
                                  : NVVM::SetMaxRegisterAction::increase;
  NVVM::SetMaxRegisterOp::create(b, b.getLoc(), adjRegs, action);
}

// Original function-local barrier lowering before the Triton #9366-style
// backport (kept for review context):
//
// // Assign hardware barriers to each warp group and rewrite warp group
// // barriers into `barrier.sync` instructions. There is a maximum number of
// // barriers.
// clang-format off
// static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
//                                               ArrayRef<WarpSpecializeOp> wsOps,
//                                               unsigned threadsPerWarp,
//                                               unsigned defaultWarpGroupSize) {
//   // HACK: Turn all `nvvm.barrier0` ops into warp group barriers.
//   func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
//     // Walk into default regions but not partition regions.
//     if (isa<WarpSpecializePartitionsOp>(op))
//       return WalkResult::skip();
//
//     if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
//       TritonLLVMIRRewriter b(bar.getLoc(), bar);
//       createBarrier(b, kDefaultWarpGroupBarrierIdx, defaultWarpGroupSize);
//       bar.erase();
//       return WalkResult::skip();
//     }
//     return WalkResult::advance();
//   });
//
//   // Each partition executes simultaneously, so each will get a different
//   // barrier ID, but note this means there is a maximum of 16 barriers.
//   for (WarpSpecializeOp op : wsOps) {
//     for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
//       unsigned barIdx = idx + kNumReservedBarriers;
//       if (barIdx >= kNumBarriers) {
//         return func.emitError("cannot support more than ")
//                << (kNumBarriers - kNumReservedBarriers)
//                << " warp group partitions";
//       }
//       unsigned warpGroupSize = threadsPerWarp * op.getPartitionNumWarps()[idx];
//       partition->walk([&](NVVM::Barrier0Op bar) {
//         TritonLLVMIRRewriter b(bar.getLoc(), bar);
//         createBarrier(b, barIdx, warpGroupSize);
//         bar.erase();
//       });
//     }
//   }
//
//   return success();
// }
// clang-format on
//
// Adapted from https://github.com/triton-lang/triton/pull/9366 (commit
// f1ece2c, "ws function calls"). The new lowering passes a hidden i32 barrier
// id through calls so barriers in outlined Triton helpers use the calling warp
// group. This NVIDIA-only FlagTree backport rewrites only helpers marked with
// ws_num_warps, preserving the ABI of internal functions emitted by TLE Raw.
static std::string mangleWarpSpecializeFunctionName(StringRef name) {
  return (name + "_ws").str();
}

static bool isInternalCall(LLVM::CallOp callOp,
                           const DenseSet<StringAttr> &internalFunctions) {
  auto callee = callOp.getCalleeAttr();
  return callee && internalFunctions.contains(callee.getAttr());
}

static void forwardWarpGroupBarrier(LLVM::CallOp callOp, Value barrier) {
  callOp.setCallee(mangleWarpSpecializeFunctionName(*callOp.getCallee()));
  callOp.getCalleeOperandsMutable().append(barrier);
}

// Rewrite barriers and calls in a warp-specialized kernel. Calls receive the
// current warp-group barrier as a hidden argument, allowing the callee to stay
// out of line while preserving the caller's warp-specialization context.
static LogicalResult
rewriteKernelWarpGroupBarriers(LLVM::LLVMFuncOp func,
                               const DenseSet<StringAttr> &internalFunctions,
                               unsigned threadsPerWarp) {
  unsigned defaultNumWarps = lookupNumWarps(func);
  SmallVector<WarpSpecializeOp> wsOps;

  WalkResult result = func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk through the default path, but handle partition regions below so
    // each one receives its own barrier id and participant count.
    if (auto partitions = dyn_cast<WarpSpecializePartitionsOp>(op)) {
      wsOps.push_back(cast<WarpSpecializeOp>(partitions->getParentOp()));
      return WalkResult::skip();
    }
    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      TritonLLVMIRRewriter b(bar.getLoc(), bar);
      createBarrier(b, kDefaultWarpGroupBarrierIdx,
                    defaultNumWarps * threadsPerWarp);
      bar.erase();
      // This is a pre-order walk, so do not visit an erased operation.
      return WalkResult::skip();
    }
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (!isInternalCall(callOp, internalFunctions))
        return WalkResult::advance();
      TritonLLVMIRRewriter b(callOp.getLoc(), callOp);
      forwardWarpGroupBarrier(callOp, b.i32_val(kDefaultWarpGroupBarrierIdx));
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Each partition executes simultaneously, so each receives a distinct
  // hardware barrier id. Calls nested in structured control flow are found by
  // the recursive region walk and receive the same id as their partition.
  for (WarpSpecializeOp wsOp : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(wsOp.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      unsigned numThreads = wsOp.getPartitionNumWarps()[idx] * threadsPerWarp;
      WalkResult partitionResult = partition->walk([&](Operation *op) {
        if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
          TritonLLVMIRRewriter b(bar.getLoc(), bar);
          createBarrier(b, barIdx, numThreads);
          bar.erase();
          // Do not continue walking an operation that has just been erased.
          // This is required even though Barrier0Op has no regions today.
          return WalkResult::skip();
        }
        if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
          if (!isInternalCall(callOp, internalFunctions))
            return WalkResult::advance();
          TritonLLVMIRRewriter b(callOp.getLoc(), callOp);
          forwardWarpGroupBarrier(callOp, b.i32_val(barIdx));
        }
        return WalkResult::advance();
      });
      if (partitionResult.wasInterrupted())
        return failure();
    }
  }

  return success();
}

static LogicalResult rewriteInnerFunctionWarpGroupBarriers(
    LLVM::LLVMFuncOp func, const DenseSet<StringAttr> &internalFunctions,
    unsigned threadsPerWarp) {
  MLIRContext *ctx = func.getContext();
  Type barrierType = IntegerType::get(ctx, 32);

  LLVM::LLVMFunctionType type = func.getFunctionType();
  SmallVector<Type> inputTypes = llvm::to_vector(type.getParams());
  inputTypes.push_back(barrierType);
  func.setFunctionType(LLVM::LLVMFunctionType::get(
      type.getReturnType(), inputTypes, type.isVarArg()));
  Value barrier = func.getBody().addArgument(barrierType, func.getLoc());

  func.setSymName(mangleWarpSpecializeFunctionName(func.getSymName()));
  if (ArrayAttr argAttrs = func.getArgAttrsAttr()) {
    SmallVector<Attribute> attrs = llvm::to_vector(argAttrs.getValue());
    attrs.push_back(DictionaryAttr::get(ctx, {}));
    func.setArgAttrsAttr(ArrayAttr::get(ctx, attrs));
  }

  auto numWarpsAttr = func->getAttrOfType<IntegerAttr>("ws_num_warps");
  if (!numWarpsAttr)
    return func.emitError("function missing 'ws_num_warps' attribute");
  unsigned numThreads = numWarpsAttr.getInt() * threadsPerWarp;

  func.walk([&](Operation *op) {
    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      TritonLLVMIRRewriter b(bar.getLoc(), bar);
      createBarrier(b, barrier, numThreads);
      bar.erase();
      return;
    }
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (isInternalCall(callOp, internalFunctions))
        forwardWarpGroupBarrier(callOp, barrier);
    }
  });

  return success();
}

// Assign hardware barriers to warp groups and preserve that assignment across
// internal function calls. This is the NVIDIA-only backport of Triton's
// function-call-aware warp-specialize barrier lowering.
static LogicalResult rewriteWarpGroupBarriers(ModuleOp module,
                                              unsigned threadsPerWarp) {
  SmallVector<LLVM::LLVMFuncOp> kernels;
  bool hasWarpSpecialize = false;
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (func.getLinkage() == LLVM::Linkage::External)
      kernels.push_back(func);

    WalkResult hasWarpSpecializeInFunc =
        func.walk([&](WarpSpecializeOp) { return WalkResult::interrupt(); });
    if (!hasWarpSpecializeInFunc.wasInterrupted())
      continue;
    if (func.getLinkage() != LLVM::Linkage::External)
      return func.emitError(
          "only top-level kernel functions can be warp-specialized");
    hasWarpSpecialize = true;
  }
  if (!hasWarpSpecialize)
    return success();

  DenseSet<StringAttr> internalFunctions;
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    // Only Triton device functions carry this attribute. TLE Raw may
    // materialize other internal LLVM helpers whose ABI must remain unchanged.
    if (func.getLinkage() != LLVM::Linkage::External &&
        func->hasAttr("ws_num_warps"))
      internalFunctions.insert(func.getSymNameAttr());
  }

  // Internal Triton helpers are renamed and gain a hidden barrier argument
  // below. Rewrite calls from every kernel in the module so a non-specialized
  // kernel cannot retain a reference to the old helper ABI.
  for (LLVM::LLVMFuncOp kernel : kernels) {
    if (failed(rewriteKernelWarpGroupBarriers(kernel, internalFunctions,
                                              threadsPerWarp)))
      return failure();
  }
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (!internalFunctions.contains(func.getSymNameAttr()))
      continue;
    if (failed(rewriteInnerFunctionWarpGroupBarriers(func, internalFunctions,
                                                     threadsPerWarp)))
      return failure();
  }

  return success();
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const NVIDIA::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  unsigned defaultNumWarps = lookupNumWarps(func);

  // Original function-local rewrite before the #9366-style backport (kept for
  // review context):
  //
  // // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  // unsigned defaultWarpGroupSize = threadsPerWarp * defaultNumWarps;
  // if (failed(rewriteWarpGroupBarriers(func, wsOps, threadsPerWarp,
  //                                     defaultWarpGroupSize)))
  //   return failure();
  //
  // Barrier rewriting now runs once at module scope before any kernel is
  // lowered, so helper signatures and every call site stay consistent.

  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
  }
  unsigned totalNumThreads = totalNumWarpsAttr.getInt() * threadsPerWarp;

  // Determine how many registers the worker warps can surrender before they
  // begin execution.
  auto maxnreg = func->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      AttrMaxRegistersName);
  int lowRegs = -1;
  int defRegs = -1;
  if (maxnreg) {
    int numWorkerWarps = totalNumWarpsAttr.getInt() - defaultNumWarps;
    int startRegs = maxnreg.getInt();

    // First determine how many extra registers the default warp group can get
    // if the workers surrender the maximum number of registers.
    lowRegs = 24;
    int extraRegs = (startRegs - lowRegs) * numWorkerWarps / defaultNumWarps;
    defRegs = (startRegs + extraRegs) / 8 * 8;

    // If the default warp group goes over 256 registers, the workers don't need
    // to give up this much.
    if (defRegs > 256) {
      defRegs = 256;
      int giveRegs = (defRegs - startRegs) * defaultNumWarps / numWorkerWarps;
      lowRegs = (startRegs - giveRegs) / 8 * 8;
    }
  }

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
#ifdef __TLE__
  tleElideTrivialCaptures(func, wsOps);
#else
  elideTrivialCaptures(func, wsOps);
#endif

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
  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  // Tell PTXAS this value is warp-uniform.
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });
#ifdef __TLE__
  hoistCtaUniformCapturesToHeader(func, wsOps, header);
#endif
  b.setInsertionPointToStart(entry);
  if (maxnreg)
    createRegRealloc(b, maxnreg.getInt(), defRegs);

  WarpSpecializeCallbacks callbacks;
  callbacks.createAllBarrier = [](TritonLLVMIRRewriter &b, unsigned barIdx) {
    createAllBarrier(b, barIdx);
  };

  callbacks.reallocRegisters = [&](TritonLLVMIRRewriter &b, WarpSpecializeOp ws,
                                   RegisterReallocPhase phase,
                                   unsigned regionNumber) {
    if (phase == RegisterReallocPhase::SwitchLoopStart) {
      if (maxnreg)
        createRegRealloc(b, maxnreg.getInt(), lowRegs);
      return;
    }

    if (auto actRegs = ws.getActualRegisters()) {
      switch (phase) {
      case RegisterReallocPhase::WorkerPartitionStart:
        createRegRealloc(b, lowRegs, (*actRegs)[regionNumber + 1]);
        break;
      case RegisterReallocPhase::WorkerPartitionEnd:
        createRegRealloc(b, (*actRegs)[regionNumber + 1], lowRegs);
        break;
      case RegisterReallocPhase::DefaultPartitionStart:
        createRegRealloc(b, defRegs, actRegs->front());
        break;
      case RegisterReallocPhase::DefaultPartitionEnd:
        createRegRealloc(b, actRegs->front(), defRegs);
        break;
      default:
        break;
      }
    }
  };

  // ^switchLoop:
  //   barrier.sync 1
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_tid = sub %tid, <default_warp_group_size>
  //   %rel_wid = udiv %rel_tid, 32

  return lowerWarpSpecializeCommon(
      func, wsOps, entry, header, switchLoop, wid, ctx, defaultNumWarps,
      totalNumWarpsAttr.getInt(), targetInfo, callbacks, kSwitchLoopBarrierIdx);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::ConvertWarpSpecializeToLLVMBase<
          ConvertWarpSpecializeToLLVM> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // FIXME: Assume warp specialization only happens on Blackwell.
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/100, /*ptxVersion=*/87);

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

    // Adapted from https://github.com/triton-lang/triton/pull/9366: rewrite
    // barrier context through outlined helper calls at module scope before
    // lowering the warp-specialize regions themselves.
    unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    if (failed(rewriteWarpGroupBarriers(mod, threadsPerWarp)))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      // Original kernel selection (kept for review context):
      // if (func.isPublic())
      //   kernels.push_back(func);
      //
      // The call-aware rewrite identifies kernels by external linkage, so use
      // the same criterion when lowering their warp-specialize regions.
      if (func.getLinkage() == LLVM::Linkage::External)
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace
