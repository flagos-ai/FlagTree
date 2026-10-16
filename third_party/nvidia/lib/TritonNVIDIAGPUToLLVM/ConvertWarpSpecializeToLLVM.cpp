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

static void createBarrier(TritonLLVMIRRewriter &b, Value barIdx,
                          unsigned numThreads) {
  // If a partition has only 1 warp, use `bar.warp.sync`.
  if (numThreads == 32)
    LLVM::NVIDIA::createSyncWarp(b.getLoc(), b);
  else
    NVVM::BarrierOp::create(b, b.getLoc(), barIdx, b.i32_val(numThreads));
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
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

#ifdef __TLE__
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
#ifdef __TLE__
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

#ifdef __TLE__
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

static void createRegRealloc(TritonLLVMIRRewriter &b, int curRegs,
                             int adjRegs) {
  curRegs = std::min(256, curRegs);
  adjRegs = std::min(256, adjRegs);
  auto action = adjRegs < curRegs ? NVVM::SetMaxRegisterAction::decrease
                                  : NVVM::SetMaxRegisterAction::increase;
  NVVM::SetMaxRegisterOp::create(b, b.getLoc(), adjRegs, action);
}

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

static void rewritePartitionRegions(WarpSpecializeOp ws, Block *switchLoop,
                                    const NVIDIA::TargetInfo &targetInfo,
                                    int lowRegs) {
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

  // ^switchLoop:
  //   barrier.sync 1
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_tid = sub %tid, <default_warp_group_size>
  //   %rel_wid = udiv %rel_tid, 32
  b.setInsertionPointToStart(switchLoop);
  if (maxnreg)
    createRegRealloc(b, maxnreg.getInt(), lowRegs);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  Value relWid = b.sub(wid, b.i32_val(defaultNumWarps));

  // The default warp group will populate the state pointer with the state ID
  // for all warps.
  // %warp_state_ptr = getelementptr ptr %state_tr[%rel_wid]
  // %warp_state = load i8 %warp_state_ptr
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

    unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    if (failed(rewriteWarpGroupBarriers(mod, threadsPerWarp)))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getLinkage() == LLVM::Linkage::External)
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace
