#include "TritonHCUGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "TritonHCUGPUToLLVM/WaitCntHCUUtility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#ifdef __TLE__
#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#endif
#include "third_party/hcu/include/Analysis/AxisInfoExt.h"
#include "third_party/hcu/include/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONHCUGPUTOLLVM
#include "TritonHCUGPUToLLVM/Passes.h.inc"
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

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
#ifdef __TLE__
    addIllegalDialect<triton::tle::TleDialect>();
#endif
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<triton::hcugpu::InstructionSchedHint>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

#ifdef __TLE__
class TleLLVMConversionTarget : public ConversionTarget {
public:
  explicit TleLLVMConversionTarget(MLIRContext &ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, ROCDL::ROCDLDialect,
                    mlir::scf::SCFDialect>();
    addIllegalDialect<triton::tle::TleDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    markUnknownOpDynamicallyLegal([](Operation *) -> bool { return true; });
  }
};
#endif

class TritonHCUGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  TritonHCUGPUToLLVMTypeConverter(MLIRContext *ctx,
                                  const LowerToLLVMOptions &options,
                                  const TargetInfoBase &targetInfo,
                                  const DataLayoutAnalysis *analysis = nullptr)
      : TritonGPUToLLVMTypeConverter(ctx, options, targetInfo, analysis) {
    addConversion([&](TensorDescType type) -> std::optional<Type> {
      return convertTensorDescType(type);
    });
  }

  Type convertTensorDescType(triton::TensorDescType type) {
    auto ctx = type.getContext();
    auto blockType = type.getBlockType();
    auto shape = blockType.getShape();

    // Determine the number of dwords based on tensor dimensions
    // 2D tensors: group0 (4) + group1 (8) = 12 dwords
    // 3D-5D tensors: group0 (4) + group1 (8) + group2 (4) + group3 (4) = 20
    // dwords
    int numDwords = (shape.size() > 2) ? (4 + 8 + 4 + 4) : (4 + 8);

    auto types = SmallVector<Type>(numDwords, IntegerType::get(ctx, 32));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
};

struct ConvertTritonHCUGPUToLLVM
    : public triton::impl::ConvertTritonHCUGPUToLLVMBase<
          ConvertTritonHCUGPUToLLVM> {
  explicit ConvertTritonHCUGPUToLLVM(StringRef targetArch, bool ftz) {
    this->arch = targetArch.str();
    this->ftz = ftz;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::hcugpu::TritonHCUGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    HCU::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == HCU::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonHCUGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);

    if (targetInfo.requiresAliasInfoForAsyncOps())
      HCU::annotateLocalLoadsSyncedViaAsyncWait(mod);

    HCU::addLocalBarrierAfterAmdGpuAsyncWait(mod);
    ModuleMembarAnalysis membarPass(&allocation,
                                    mlir::triton::HCU::membarFilter);
    membarPass.run();

    AMD::addWaitCntHCU(mod);

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

    HCU::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

#ifdef __TLE__
    {
      TleLLVMConversionTarget tleTarget(*context, typeConverter);
      RewritePatternSet tlePatterns(context);
      mlir::triton::tle::populateExtractTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateInsertTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateLocalPointersOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns,
          patternBenefitPrioritizeOverLLVMConversions);
      if (failed(
              applyPartialConversion(mod, tleTarget, std::move(tlePatterns))))
        return signalPassFailure();
    }
#endif

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for HCU specific patterns higher so they apply before common
    // patterns
    int HCUBenefit = commonBenefit + 1;
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

    HCU::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, HCUBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    HCU::populateDotOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                     HCUBenefit);
    HCU::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                             axisInfoAnalysis, allocation,
                                             targetInfo, HCUBenefit);
    HCU::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, HCUBenefit);
    HCU::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    HCU::populateBarrierOpToLLVMPatterns(typeConverter, patterns, HCUBenefit);
    HCU::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                            HCUBenefit);
    HCUBenefit += 1;
    HCU::populateMLSOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                     axisInfoAnalysis, HCUBenefit);

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

    HCU::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        HCUBenefit);
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
    HCU::populateSPMDOpToLLVMPattern(typeConverter, patterns, HCUBenefit);

    mlir::triton::HCU::populateTritonHCUGPUToLLVMPatterns(typeConverter,
                                                          patterns, HCUBenefit);
    mlir::triton::HCU::populateUpcastMXFPToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo, HCUBenefit);
    mlir::triton::HCU::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, HCUBenefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->arch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid HCUGPU chipset name: " + this->arch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    HCU::adjustModeRegister(mod, targetInfo);
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
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonHCUGPUToLLVMPass(StringRef targetArch, bool ftz) {
  return std::make_unique<ConvertTritonHCUGPUToLLVM>(targetArch, ftz);
}

} // namespace mlir::triton
