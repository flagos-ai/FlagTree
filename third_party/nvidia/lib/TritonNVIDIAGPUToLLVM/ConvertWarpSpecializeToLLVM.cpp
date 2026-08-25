#ifdef __TLE__
#include <cstdint>
#include <optional>
#include <string>
#endif

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
#include "llvm/ADT/DenseMap.h"
#ifdef __TLE__
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#endif

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

static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
                          unsigned numThreads) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  // If a partition has only 1 warp, use `bar.warp.sync`.
  if (numThreads == 32)
    LLVM::NVIDIA::createSyncWarp(b.getLoc(), b);
  else
    NVVM::BarrierOp::create(b, b.getLoc(), b.i32_val(barIdx),
                            b.i32_val(numThreads));
}

static void createAllBarrier(TritonLLVMIRRewriter &b, unsigned barIdx) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  LLVM::createLLVMIntrinsicCallOp(b, b.getLoc(),
                                  "llvm.nvvm.barrier.cta.sync.all",
                                  void_ty(b.getContext()), b.i32_val(barIdx));
}

#ifdef __TLE__
static Value createWorkerRelativeWarpId(TritonLLVMIRRewriter &b,
                                        unsigned defaultNumWarps) {
  std::string asmString =
      "mov.u32 $0, %tid.x;\n\tshr.u32 $0, $0, 5;\n\tsub.u32 $0, $0, " +
      std::to_string(defaultNumWarps) + ";";
  return LLVM::InlineAsmOp::create(
             b, b.getLoc(), b.getI32Type(), ValueRange{}, asmString, "=r",
             /*has_side_effects=*/true,
             /*is_align_stack=*/false, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(b.getContext(),
                                       LLVM::AsmDialect::AD_ATT),
             ArrayAttr::get(b.getContext(), {}))
      .getResult(0);
}
#endif

namespace {

struct BarrierExecutionScope {
  enum class Kind {
    Cta,
    WarpGroup,
  };

  Kind kind;
  unsigned barrierIndex = 0;
  unsigned numThreads = 0;

  static BarrierExecutionScope cta() {
    return {Kind::Cta, /*barrierIndex=*/0, /*numThreads=*/0};
  }

  static BarrierExecutionScope warpGroup(unsigned barrierIndex,
                                         unsigned numThreads) {
    return {Kind::WarpGroup, barrierIndex, numThreads};
  }

  bool isEquivalentTo(const BarrierExecutionScope &other) const {
    if (kind != other.kind)
      return false;
    if (kind == Kind::Cta)
      return true;
    // One-warp barriers lower to `bar.warp.sync`; their hardware barrier ID is
    // therefore not part of the execution contract.
    if (numThreads == 32 && other.numThreads == 32)
      return true;
    return barrierIndex == other.barrierIndex &&
           numThreads == other.numThreads;
  }
};

using BarrierScopeMap =
    llvm::DenseMap<Operation *, SmallVector<BarrierExecutionScope, 2>>;

struct ScopedFunction {
  LLVM::LLVMFuncOp function;
  BarrierExecutionScope scope;
};

static bool addBarrierScope(BarrierScopeMap &scopes, LLVM::LLVMFuncOp function,
                            BarrierExecutionScope scope) {
  auto &functionScopes = scopes[function.getOperation()];
  if (llvm::any_of(functionScopes, [&](const BarrierExecutionScope &existing) {
        return existing.isEquivalentTo(scope);
      }))
    return false;
  functionScopes.push_back(scope);
  return true;
}

static LogicalResult
enqueueScopedCallee(LLVM::CallOp call, BarrierExecutionScope scope,
                    ModuleOp module, BarrierScopeMap &scopes,
                    SmallVectorImpl<ScopedFunction> &worklist) {
  std::optional<StringRef> calleeName = call.getCallee();
  if (!calleeName &&
      scope.kind == BarrierExecutionScope::Kind::WarpGroup)
    return call.emitError(
        "cannot determine warp-group barrier scope across an indirect call");
  if (!calleeName)
    return success();

  auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*calleeName);
  // External functions have no body in which a CTA barrier can occur.
  if (!callee || callee.getBody().empty())
    return success();

  bool containsWarpSpecialize = false;
  callee.walk([&](WarpSpecializeOp) { containsWarpSpecialize = true; });
  if (containsWarpSpecialize)
    return call.emitError("a warp-specialized callee is unsupported; "
                          "warp specialization must be rooted at a kernel");

  if (addBarrierScope(scopes, callee, scope))
    worklist.push_back({callee, scope});
  return success();
}

static LogicalResult
enqueueCallsInDefaultWarpGroup(LLVM::LLVMFuncOp kernel,
                               BarrierExecutionScope scope, ModuleOp module,
                               BarrierScopeMap &scopes,
                               SmallVectorImpl<ScopedFunction> &worklist) {
  LogicalResult result = success();
  kernel.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Partition regions execute under their own barrier scopes.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();
    if (auto call = dyn_cast<LLVM::CallOp>(op)) {
      if (failed(
              enqueueScopedCallee(call, scope, module, scopes, worklist))) {
        result = failure();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

static LogicalResult
collectTransitiveBarrierScopes(ModuleOp module,
                               ArrayRef<LLVM::LLVMFuncOp> kernels,
                               unsigned threadsPerWarp,
                               BarrierScopeMap &scopes) {
  SmallVector<ScopedFunction> worklist;

  for (LLVM::LLVMFuncOp kernel : kernels) {
    SmallVector<WarpSpecializeOp> wsOps;
    kernel.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });

    if (wsOps.empty()) {
      if (failed(enqueueCallsInDefaultWarpGroup(
              kernel, BarrierExecutionScope::cta(), module, scopes, worklist)))
        return failure();
      continue;
    }

    unsigned defaultWarpGroupSize =
        threadsPerWarp * lookupNumWarps(kernel);
    if (failed(enqueueCallsInDefaultWarpGroup(
            kernel,
            BarrierExecutionScope::warpGroup(
                kDefaultWarpGroupBarrierIdx, defaultWarpGroupSize),
            module, scopes, worklist)))
      return failure();

    for (WarpSpecializeOp wsOp : wsOps) {
      for (auto [idx, partition] :
           llvm::enumerate(wsOp.getPartitionRegions())) {
        unsigned barrierIndex = idx + kNumReservedBarriers;
        if (barrierIndex >= kNumBarriers)
          return kernel.emitError("cannot support more than ")
                 << (kNumBarriers - kNumReservedBarriers)
                 << " warp group partitions";
        unsigned partitionSize =
            threadsPerWarp * wsOp.getPartitionNumWarps()[idx];
        BarrierExecutionScope scope =
            BarrierExecutionScope::warpGroup(barrierIndex, partitionSize);
        LogicalResult result = success();
        partition->walk([&](LLVM::CallOp call) {
          if (failed(
                  enqueueScopedCallee(call, scope, module, scopes, worklist)))
            result = failure();
        });
        if (failed(result))
          return failure();
      }
    }
  }

  while (!worklist.empty()) {
    ScopedFunction scopedFunction = worklist.pop_back_val();
    LogicalResult result = success();
    scopedFunction.function.walk([&](LLVM::CallOp call) {
      if (failed(enqueueScopedCallee(call, scopedFunction.scope, module, scopes,
                                     worklist)))
        result = failure();
    });
    if (failed(result))
      return failure();
  }

  return success();
}

static LogicalResult
rewriteTransitiveWarpGroupBarriers(ModuleOp module,
                                   ArrayRef<LLVM::LLVMFuncOp> kernels,
                                   unsigned threadsPerWarp) {
  BarrierScopeMap scopes;
  if (failed(collectTransitiveBarrierScopes(module, kernels, threadsPerWarp,
                                            scopes)))
    return failure();

  for (auto &[operation, functionScopes] : scopes) {
    auto function = cast<LLVM::LLVMFuncOp>(operation);
    SmallVector<NVVM::Barrier0Op> barriers;
    function.walk([&](NVVM::Barrier0Op barrier) {
      barriers.push_back(barrier);
    });
    if (barriers.empty())
      continue;

    if (functionScopes.size() != 1) {
      InFlightDiagnostic diagnostic = function.emitError(
          "barrier-bearing function is called from incompatible execution "
          "scopes");
      for (const BarrierExecutionScope &scope : functionScopes) {
        if (scope.kind == BarrierExecutionScope::Kind::Cta) {
          diagnostic.attachNote(function.getLoc()) << "called from CTA scope";
        } else {
          diagnostic.attachNote(function.getLoc())
              << "called from warp-group scope with barrier "
              << scope.barrierIndex << " and " << scope.numThreads
              << " participating threads";
        }
      }
      return failure();
    }

    const BarrierExecutionScope &scope = functionScopes.front();
    if (scope.kind == BarrierExecutionScope::Kind::Cta)
      continue;

    for (NVVM::Barrier0Op barrier : barriers) {
      TritonLLVMIRRewriter rewriter(barrier.getLoc(), barrier);
      createBarrier(rewriter, scope.barrierIndex, scope.numThreads);
      barrier.erase();
    }
  }

  return success();
}

} // namespace

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

static std::optional<int32_t>
getKernelArgumentTableOffset(LLVM::LLVMFuncOp func, BlockArgument arg) {
  if (arg.getOwner() != &func.getBody().front())
    return std::nullopt;

  auto offsets = func->getAttrOfType<DenseI32ArrayAttr>(
      tle::kTleWarpSpecializeKernelArgumentTableOffsetsAttr);
  if (!offsets || arg.getArgNumber() >= offsets.size())
    return std::nullopt;
  int32_t offset = offsets.asArrayRef()[arg.getArgNumber()];
  if (offset < 0)
    return std::nullopt;
  return offset;
}

static Value createKernelArgumentReload(
    LLVM::LLVMFuncOp func, Type type, int32_t byteOffset,
    const NVIDIA::TargetInfo &targetInfo, Operation *user) {
  TritonLLVMIRRewriter b(user->getLoc(), user);
  Value tableBase =
      LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
  Value ptr = b.gep(ptrTy, b.getI8Type(), tableBase,
                    LLVM::GEPArg(byteOffset));
  LLVM::LoadOp reload = b.load(type, ptr, /*align=*/8);
  reload.setVolatile_(true);
  return reload;
}

static void reloadKernelArgumentsInWarpSpecializeRegions(
    LLVM::LLVMFuncOp func, const NVIDIA::TargetInfo &targetInfo) {
  for (BlockArgument arg : func.getArguments()) {
    std::optional<int32_t> tableOffset =
        getKernelArgumentTableOffset(func, arg);
    if (!tableOffset)
      continue;

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : arg.getUses())
      uses.push_back(&use);
    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      if (isa<WarpSpecializeOp>(user) ||
          !user->getParentOfType<WarpSpecializeOp>())
        continue;
      Value reload = createKernelArgumentReload(
          func, arg.getType(), *tableOffset, targetInfo, user);
      use->set(reload);
    }
  }
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
      if (arg.getOwner() == &func.getBody().front()) {
        continue;
      }
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

static LogicalResult
elideTrivialCaptures(LLVM::LLVMFuncOp func,
                     ArrayRef<WarpSpecializeOp> wsOps,
                     const NVIDIA::TargetInfo &targetInfo) {
  // The goal is to completely eliminate captures by hoisting or rematerializing
  // computations. We could minimize captures by rematerializing
  // subcomputations, but that is much more complicated. Prefer rematerializing
  // because that reduces liveranges. If subgraphs are duplicated more than
  // once, we will rely on CSE to clean them up.
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    llvm::BitVector toErase(wsOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
#ifdef __TLE__
      if (auto arg = dyn_cast<BlockArgument>(capture)) {
        if (auto tableOffset = getKernelArgumentTableOffset(func, arg)) {
          toErase.set(i);
          for (Region *region : wsOp.getPartitionRegions()) {
            BlockArgument partitionArg = region->getArgument(i);
            SmallVector<OpOperand *> uses;
            for (OpOperand &use : partitionArg.getUses())
              uses.push_back(&use);

            for (OpOperand *use : uses) {
              Operation *user = use->getOwner();
              Value reload = createKernelArgumentReload(
                  func, partitionArg.getType(), *tableOffset, targetInfo,
                  user);
              use->set(reload);
            }
          }
          continue;
        }
      }
#endif
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
  return success();
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
      if (arg.getOwner() == &func.getBody().front()) {
        if (getKernelArgumentTableOffset(func, arg))
          return failure();
        continue;
      }
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

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into `barrier.sync` instructions. There is a maximum number of barriers.
static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                              ArrayRef<WarpSpecializeOp> wsOps,
                                              unsigned threadsPerWarp,
                                              unsigned defaultWarpGroupSize) {
  // HACK: Turn all `nvvm.barrier0` ops into warp group barriers.
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      TritonLLVMIRRewriter b(bar.getLoc(), bar);
      createBarrier(b, kDefaultWarpGroupBarrierIdx, defaultWarpGroupSize);
      bar.erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

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
      partition->walk([&](NVVM::Barrier0Op bar) {
        TritonLLVMIRRewriter b(bar.getLoc(), bar);
        createBarrier(b, barIdx, warpGroupSize);
        bar.erase();
      });
    }
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

static bool isTrivialStaticPartition(Region *partition) {
  if (!partition->hasOneBlock())
    return false;
  return llvm::all_of(partition->front(),
                      [](Operation &op) { return isa<WarpReturnOp>(op); });
}

static void rewriteStaticPartitionRegions(
    WarpSpecializeOp ws, Block *after,
    const NVIDIA::TargetInfo &targetInfo, int lowRegs) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());

  for (Region *partition : ws.getPartitionRegions()) {
    b.setInsertionPointToStart(&partition->front());

    if (auto actRegs = ws.getActualRegisters()) {
      int partitionRegs =
          (*actRegs)[partition->getRegionNumber() + 1];
      if (partitionRegs != lowRegs)
        createRegRealloc(b, lowRegs, partitionRegs);
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
        Value value = b.load(arg.getType(), ptr, /*align=*/1);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The default region may reuse the capture allocation as soon as it
    // starts. Ensure every worker has loaded its copy first. Capture-free
    // one-shot regions do not need this second rendezvous.
    if (ws.getNumOperands())
      createAllBarrier(b, kSwitchLoopBarrierIdx);

    partition->walk([&](WarpReturnOp op) {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, after);
    });
  }
}

static LogicalResult lowerStaticOneShotWarpSpecialize(
    LLVM::LLVMFuncOp func, WarpSpecializeOp ws,
    const NVIDIA::TargetInfo &targetInfo, unsigned threadsPerWarp,
    unsigned defaultNumWarps, IntegerAttr maxnreg,
    bool usesDynamicRegisterReallocation, int lowRegs, int defRegs) {
  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  SmallVector<Region *> partitions =
      llvm::to_vector(ws.getPartitionRegions());
  SmallVector<Block *> partitionEntries;
  SmallVector<bool> trivialPartitions;
  for (Region *partition : partitions) {
    partitionEntries.push_back(&partition->front());
    trivialPartitions.push_back(isTrivialStaticPartition(partition));
  }

  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
  Block *roleDispatch = b.createBlock(entry);
  Block *workerPrelude = b.createBlock(entry);
  Block *defaultRendezvous = b.createBlock(entry);
  Block *workerRendezvous = b.createBlock(entry);
#ifdef __TLE__
  Block *kernelArgumentInit = nullptr;
  auto kernelArgumentTableOffsets = func->getAttrOfType<DenseI32ArrayAttr>(
      tle::kTleWarpSpecializeKernelArgumentTableOffsetsAttr);
  if (kernelArgumentTableOffsets)
    kernelArgumentInit = b.createBlock(roleDispatch);
#endif

  b.setInsertionPointToStart(header);
  Value tid = NVVM::ThreadIdXOp::create(b, b.getLoc(), i32_ty);
  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
#ifdef __TLE__
  if (kernelArgumentInit) {
    Value isThreadZero = b.icmp_eq(tid, b.i32_val(0));
    LLVM::CondBrOp::create(b, b.getLoc(), isThreadZero, kernelArgumentInit,
                           roleDispatch);

    b.setInsertionPointToStart(kernelArgumentInit);
    Value tableBase =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);
    for (auto [i, byteOffset] :
         llvm::enumerate(kernelArgumentTableOffsets.asArrayRef())) {
      if (byteOffset < 0)
        continue;
      Value ptr =
          b.gep(ptrTy, i8_ty, tableBase, LLVM::GEPArg(byteOffset));
      b.store(header->getArgument(i), ptr, /*align=*/8);
    }
    LLVM::BrOp::create(b, b.getLoc(), roleDispatch);
  } else {
#endif
    LLVM::BrOp::create(b, b.getLoc(), roleDispatch);
#ifdef __TLE__
  }
#endif

  b.setInsertionPointToStart(roleDispatch);
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, workerPrelude);

  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });
#ifdef __TLE__
  hoistCtaUniformCapturesToHeader(func, {ws}, header);
#endif

  Operation *precedingBarrier = ws->getPrevNode();
  if (isa_and_nonnull<NVVM::BarrierOp, NVVM::Barrier0Op>(precedingBarrier))
    precedingBarrier->erase();

  Block *before = ws->getBlock();
  Block *after = b.splitBlock(before, ws->getIterator());

  b.setInsertionPointToEnd(before);
  if (ws.getNumOperands()) {
    auto captureType = LLVM::LLVMStructType::getLiteral(
        b.getContext(), llvm::to_vector(ws.getOperandTypes()),
        /*isPacked=*/true);
    Value capturePtr =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
    LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);
    for (auto [i, capture] : llvm::enumerate(ws.getExplicitCaptures())) {
      Value ptr =
          b.gep(ptrTy, captureType, capturePtr,
                ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(i)});
      b.store(capture, ptr, /*align=*/1);
    }
  }
  LLVM::BrOp::create(b, b.getLoc(), defaultRendezvous);

  b.setInsertionPointToStart(workerPrelude);
  if (usesDynamicRegisterReallocation && lowRegs != maxnreg.getInt())
    createRegRealloc(b, maxnreg.getInt(), lowRegs);
  LLVM::BrOp::create(b, b.getLoc(), workerRendezvous);

  Block *defaultEntry = &ws.getDefaultRegion().front();
  b.setInsertionPointToStart(defaultEntry);
  if (usesDynamicRegisterReallocation && defRegs != maxnreg.getInt())
    createRegRealloc(b, maxnreg.getInt(), defRegs);

  ws.getDefaultRegion().walk([&](WarpYieldOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    b.replaceOpWithNewOp<LLVM::BrOp>(op, after);
  });
  rewriteStaticPartitionRegions(ws, after, targetInfo, lowRegs);

  Region::BlockListType &funcBlocks = func.getBody().getBlocks();
  funcBlocks.splice(after->getIterator(), ws.getDefaultRegion().getBlocks());
  for (Region *partition : partitions)
    funcBlocks.splice(after->getIterator(), partition->getBlocks());

  Block *workerTarget = after;
  ArrayRef<int32_t> startIds = *ws.getWarpGroupStartIds();
  ArrayRef<int32_t> numWarps = ws.getPartitionNumWarps();
  for (int i = static_cast<int>(partitions.size()) - 1; i >= 0; --i) {
    // A padding partition can be skipped only when there is no capture
    // lifetime rendezvous for its warps to participate in.
    if (trivialPartitions[i] && ws.getNumOperands() == 0)
      continue;
    Block *test = b.createBlock(after);
    b.setInsertionPointToStart(test);
    Value relativeWid = b.sub(wid, b.i32_val(startIds[i]));
    Value isPartition = b.icmp_ult(relativeWid, b.i32_val(numWarps[i]));
    LLVM::CondBrOp::create(b, b.getLoc(), isPartition, partitionEntries[i],
                           workerTarget);
    workerTarget = test;
  }

  // Keep the two control-flow paths separate across the rendezvous. Values
  // computed by the default prelude are intentionally unavailable to worker
  // warps, so merging the paths before entering the default region would
  // break SSA dominance. Both paths still arrive at the same named CTA
  // barrier and therefore form one hardware rendezvous.
  b.setInsertionPointToStart(defaultRendezvous);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  if (ws.getNumOperands())
    createAllBarrier(b, kSwitchLoopBarrierIdx);
  LLVM::BrOp::create(b, b.getLoc(), defaultEntry);

  b.setInsertionPointToStart(workerRendezvous);
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  LLVM::BrOp::create(b, b.getLoc(), workerTarget);

  ws.erase();
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

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const NVIDIA::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  bool useStaticOneShotLowering =
      wsOps.size() == 1 &&
      wsOps.front()->hasAttr(kStaticWarpRolesAttrName) &&
      isStaticOneShotWarpSpecialize(wsOps.front());

  // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  unsigned defaultNumWarps = lookupNumWarps(func);
  unsigned defaultWarpGroupSize = threadsPerWarp * defaultNumWarps;
  if (failed(rewriteWarpGroupBarriers(func, wsOps, threadsPerWarp,
                                      defaultWarpGroupSize)))
    return failure();

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
  bool usesDynamicRegisterReallocation = llvm::any_of(
      wsOps, [](WarpSpecializeOp ws) {
        return ws.getActualRegisters().has_value();
      });
  int lowRegs = -1;
  int defRegs = -1;
  if (maxnreg && usesDynamicRegisterReallocation) {
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
  if (failed(elideTrivialCaptures(func, wsOps, targetInfo)))
    return failure();
#ifdef __TLE__
  reloadKernelArgumentsInWarpSpecializeRegions(func, targetInfo);
#endif

  if (useStaticOneShotLowering) {
    return lowerStaticOneShotWarpSpecialize(
        func, wsOps.front(), targetInfo, threadsPerWarp, defaultNumWarps,
        maxnreg, usesDynamicRegisterReallocation, lowRegs, defRegs);
  }

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
#ifdef __TLE__
  Block *kernelArgumentInit = nullptr;
  Block *dispatch = nullptr;
  auto kernelArgumentTableOffsets = func->getAttrOfType<DenseI32ArrayAttr>(
      tle::kTleWarpSpecializeKernelArgumentTableOffsetsAttr);
  if (kernelArgumentTableOffsets) {
    kernelArgumentInit = b.createBlock(entry);
    dispatch = b.createBlock(entry);
  }
#endif
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = NVVM::ThreadIdXOp::create(b, b.getLoc(), i32_ty);
  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  // Tell PTXAS this value is warp-uniform.
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
#ifdef __TLE__
  if (kernelArgumentInit) {
    Value isThreadZero = b.icmp_eq(tid, b.i32_val(0));
    LLVM::CondBrOp::create(b, b.getLoc(), isThreadZero, kernelArgumentInit,
                           dispatch);

    b.setInsertionPointToStart(kernelArgumentInit);
    Value tableBase =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);
    for (auto [i, byteOffset] :
         llvm::enumerate(kernelArgumentTableOffsets.asArrayRef())) {
      if (byteOffset < 0)
        continue;
      Value ptr =
          b.gep(ptrTy, i8_ty, tableBase, LLVM::GEPArg(byteOffset));
      b.store(header->getArgument(i), ptr, /*align=*/8);
    }
    LLVM::BrOp::create(b, b.getLoc(), dispatch);

    b.setInsertionPointToStart(dispatch);
    createAllBarrier(b, kSwitchLoopBarrierIdx);
    LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);
  } else {
#endif
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);
#ifdef __TLE__
  }
#endif

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });
#ifdef __TLE__
  hoistCtaUniformCapturesToHeader(func, wsOps, header);
#endif
  b.setInsertionPointToStart(entry);
  if (usesDynamicRegisterReallocation)
    createRegRealloc(b, maxnreg.getInt(), defRegs);

  // ^switchLoop:
  //   barrier.sync 1
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_tid = sub %tid, <default_warp_group_size>
  //   %rel_wid = udiv %rel_tid, 32
  b.setInsertionPointToStart(switchLoop);
  if (usesDynamicRegisterReallocation)
    createRegRealloc(b, maxnreg.getInt(), lowRegs);
#ifdef __TLE__
  Value relWid = usesDynamicRegisterReallocation
                     ? createWorkerRelativeWarpId(b, defaultNumWarps)
                     : b.sub(wid, b.i32_val(defaultNumWarps));
#endif
  createAllBarrier(b, kSwitchLoopBarrierIdx);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
#ifndef __TLE__
  Value relWid = b.sub(wid, b.i32_val(defaultNumWarps));
#endif

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

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }
    unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    if (failed(
            rewriteTransitiveWarpGroupBarriers(mod, kernels, threadsPerWarp)))
      return signalPassFailure();
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace
