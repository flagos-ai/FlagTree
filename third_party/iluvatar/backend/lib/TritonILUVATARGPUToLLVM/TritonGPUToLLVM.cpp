#include "TritonILUVATARGPUToLLVM/Passes.h"

#ifdef __ILUVATAR_TLE__
#include "Conversion/TleToLLVM.h"
#include "Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"
#endif
#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonILUVATARGPUToLLVM/MembarUtility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#ifdef __ILUVATAR_TLE__
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#endif
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <cstdlib>
#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONILUVATARGPUTOLLVM
#include "TritonILUVATARGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
#ifdef __ILUVATAR_TLE__
    mlir::triton::iluvatar_tle::addIllegalDialects(*this);
    addIllegalDialect<triton::tle::TleDialect>();
#endif
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

#ifdef __ILUVATAR_TLE__
// MembarAnalysis inserts the SME async-wait rendezvous immediately after each
// branch-local ttg.async_wait. SCF has already been converted to CFG when this
// pass runs. When both successor blocks of a cf.cond_br end in the same
// barrier.alu and branch to one common block, keeping two copies creates two
// hardware rendezvous program points even though both paths reconverge. Factor
// the barrier into the common successor so every path still executes it once.
static void coalesceSmeIfAluBarriers(ModuleOp mod) {
  SmallVector<cf::CondBranchOp> candidates;
  mod.walk([&](cf::CondBranchOp cond) {
    Block *thenBlock = cond.getTrueDest();
    Block *elseBlock = cond.getFalseDest();
    auto thenBranch = dyn_cast<cf::BranchOp>(thenBlock->getTerminator());
    auto elseBranch = dyn_cast<cf::BranchOp>(elseBlock->getTerminator());
    if (!thenBranch || !elseBranch ||
        thenBranch.getDest() != elseBranch.getDest())
      return;
    Block *commonBlock = thenBranch.getDest();
    SmallVector<Block *> predecessors(commonBlock->getPredecessors());
    if (predecessors.size() != 2 ||
        !llvm::is_contained(predecessors, thenBlock) ||
        !llvm::is_contained(predecessors, elseBlock))
      return;
    Operation *thenBarrier = thenBlock->getTerminator()->getPrevNode();
    Operation *elseBarrier = elseBlock->getTerminator()->getPrevNode();
    auto isAluBarrier = [](Operation *op) {
      auto call = dyn_cast_or_null<LLVM::CallIntrinsicOp>(op);
      return call && call.getIntrin() == "llvm.bi.sl.barrier.alu";
    };
    if (isAluBarrier(thenBarrier) && isAluBarrier(elseBarrier))
      candidates.push_back(cond);
  });

  for (cf::CondBranchOp cond : candidates) {
    Block *thenBlock = cond.getTrueDest();
    Block *elseBlock = cond.getFalseDest();
    auto thenBranch = cast<cf::BranchOp>(thenBlock->getTerminator());
    Block *commonBlock = thenBranch.getDest();
    Operation *thenBarrier = thenBlock->getTerminator()->getPrevNode();
    Operation *elseBarrier = elseBlock->getTerminator()->getPrevNode();
    thenBarrier->moveBefore(&commonBlock->front());
    elseBarrier->erase();
  }
}

// ivcore11 exposes only a transaction-level G2S wait counter.  Preserve the
// width of a statically uniform SME producer group so AsyncWaitOpConversion
// can leave one group outstanding without using ivcore40-only wait-commit.
static std::optional<unsigned>
getIluvatarSmeTransactionsPerGroup(triton::gpu::AsyncWaitOp wait) {
  auto func = wait->getParentOfType<triton::FuncOp>();
  if (!func)
    return std::nullopt;

  std::optional<unsigned> transactions;
  bool sawSmeCopy = false;
  bool valid = true;
  func.walk([&](triton::gpu::AsyncCopyGlobalToLocalOp copy) {
    if (!copy.isIluvatarSmeAsyncCopy())
      return;
    sawSmeCopy = true;
    if (!valid)
      return;
    auto dstTy = dyn_cast<triton::gpu::MemDescType>(copy.getResult().getType());
    auto enc = copy->getAttrOfType<triton::gpu::BlockedEncodingAttr>(
        "tle.explicit_memory_encoding");
    if (!enc && dstTy)
      enc = dyn_cast<triton::gpu::BlockedEncodingAttr>(dstTy.getEncoding());
    if (!dstTy || !enc || !enc.getIsSme()) {
      valid = false;
      return;
    }

    unsigned bitwidth = dstTy.getElementType().getIntOrFloatBitWidth();
    auto shape = dstTy.getShape();
    auto order = enc.getOrder();
    if (shape.size() != 2 || order.size() != 2 ||
        (bitwidth != 8 && bitwidth != 16 && bitwidth != 32)) {
      valid = false;
      return;
    }
    unsigned elemBytes = bitwidth / 8;
    bool isRowMajor = order[0] != 0;
    unsigned tileRows = 16;
    if (isRowMajor && bitwidth == 16 && shape[0] >= 4 && shape[0] < 16)
      tileRows = 4;
    auto warps = enc.getSmeWarpsPerCTA();
    if (warps.size() != 2) {
      valid = false;
      return;
    }

    unsigned offset0 = isRowMajor ? tileRows : 64 / elemBytes;
    unsigned offset1 = isRowMajor ? 64 / elemBytes : tileRows;
    unsigned ctaRows = warps[0] * offset0;
    unsigned ctaCols = warps[1] * offset1;
    if (ctaRows == 0 || ctaCols == 0 || shape[0] % ctaRows != 0 ||
        shape[1] % ctaCols != 0) {
      valid = false;
      return;
    }
    unsigned count = (shape[0] / ctaRows) * (shape[1] / ctaCols);
    if (!count || (transactions && *transactions != count)) {
      valid = false;
      return;
    }
    transactions = count;
  });
  if (!sawSmeCopy || !valid)
    return std::nullopt;
  return transactions;
}

static void annotateIluvatarSmeAsyncWaitTransactions(ModuleOp mod) {
  mod.walk([&](triton::gpu::AsyncWaitOp wait) {
    if (!wait->hasAttr("tle.explicit_async_wait") || wait.getNum() <= 0)
      return;
    auto transactions = getIluvatarSmeTransactionsPerGroup(wait);
    if (transactions) {
      wait->setAttr(
          "iluvatar.g2s_transactions_per_group",
          IntegerAttr::get(IntegerType::get(mod.getContext(), 32),
                           *transactions));
    }
  });
}
#endif

struct ConvertTritonILUVATARGPUToLLVM
    : public triton::impl::ConvertTritonILUVATARGPUToLLVMBase<
          ConvertTritonILUVATARGPUToLLVM> {
  explicit ConvertTritonILUVATARGPUToLLVM(StringRef targetArch, bool ftz,
                                          bool disableLoadVectorize) {
    this->arch = targetArch.str();
    this->ftz = ftz;
    this->disableLoadVectorize = disableLoadVectorize;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::iluvatargpu::TritonILUVATARGPUDialect>();
#ifdef __ILUVATAR_TLE__
    mlir::triton::iluvatar_tle::registerDialects(registry);
    registry.insert<mlir::triton::tle::TleDialect>();
#endif
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ILUVATAR::TargetInfo targetInfo(arch.getValue());

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // preprocess
    decomposeSmeLoadOp(
        mod); // change resultType to ptr in order to pass ABase.x
#ifdef __ILUVATAR_TLE__
    if (targetInfo.getArch() == "ivcore11")
      annotateIluvatarSmeAsyncWaitTransactions(mod);
#endif

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);

    ModuleMembarAnalysis membarPass(&allocation,
                                    triton::ILUVATAR::membarFilter);
    membarPass.run();
#ifdef __ILUVATAR_TLE__
    if (targetInfo.getArch() == "ivcore11")
      coalesceSmeIfAluBarriers(mod);
#endif

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for ILUVATAR specific patterns higher so they apply before
    // common patterns
    int ILUVATARBenefit = commonBenefit + 1;
    auto populatePatterns1 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, axisInfoAnalysis, allocation,
                   benefit);
    };

    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns6 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, axisInfoAnalysis, allocation,
                   targetInfo, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

#ifdef __ILUVATAR_TLE__
    mlir::triton::iluvatar_tle::populateTleToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    mlir::triton::tle::populateDistributedBarrierOpToLLVMPatterns(
        typeConverter, patterns, commonBenefit);
#endif
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    ILUVATAR::populateDotOpToLLVMPatterns(typeConverter, patterns,
                                          axisInfoAnalysis, ILUVATARBenefit);
    ILUVATAR::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                                  axisInfoAnalysis, allocation,
                                                  targetInfo, ILUVATARBenefit);
    ILUVATAR::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, ILUVATARBenefit,
        disableLoadVectorize);
    ILUVATAR::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    ILUVATAR::populateBarrierOpToLLVMPatterns(typeConverter, patterns,
                                              ILUVATARBenefit, targetInfo);
    // ILUVATAR::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
    //                                         ILUVATARBenefit);

    populatePatterns7(mlir::triton::populateReduceOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateGatherOpToLLVMPatterns,
                      commonBenefit);

    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    ILUVATAR::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                          ILUVATARBenefit);

    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Equivalent layouts can still leave materialization casts because dialect
    // conversion bridges original tensor SSA values to their LLVM struct form.
    // Fold those cycles before exporting LLVM dialect to LLVM IR.
    SmallVector<UnrealizedConversionCastOp> unrealizedCasts;
    mod.walk(
        [&](UnrealizedConversionCastOp op) { unrealizedCasts.push_back(op); });
    reconcileUnrealizedCasts(unrealizedCasts);

    fixUpLoopAnnotation(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }

  void decomposeSmeLoadOp(ModuleOp mod) const {
    mod.walk([&](triton::LoadOp loadOp) -> void {
      OpBuilder builder(loadOp);
      auto ptr = loadOp.getPtr();
      auto ptrTy = mlir::dyn_cast<RankedTensorType>(ptr.getType());
      if (!ptrTy)
        return;
      auto ptrBlocked =
          mlir::dyn_cast<triton::gpu::BlockedEncodingAttr>(ptrTy.getEncoding());
      if (!ptrBlocked || !ptrBlocked.getIsSme()) {
        return;
      }
      auto newRetType = RankedTensorType::get(
          ptrTy.getShape(), ptrTy.getElementType(), ptrBlocked);
      auto newload = triton::LoadOp::create(
          builder, loadOp.getLoc(), newRetType, ptr, loadOp.getMask(),
          loadOp.getOther(), loadOp.getBoundaryCheckAttr(),
          loadOp.getPaddingAttr(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile(), loadOp.getInputStride());
      loadOp.replaceAllUsesWith(newload.getResult());
      loadOp.erase();
    });
  }

  static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                              Type promotedType) {
    Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                  .cloneWith(std::nullopt, promotedType);
    return triton::FpToFpOp::create(builder, loc, tensorPromotedType, operand);
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonILUVATARGPUToLLVMPass(StringRef targetArch, bool ftz,
                                        bool disableLoadVectorize) {
  return std::make_unique<ConvertTritonILUVATARGPUToLLVM>(targetArch, ftz,
                                                          disableLoadVectorize);
}

} // namespace mlir::triton
