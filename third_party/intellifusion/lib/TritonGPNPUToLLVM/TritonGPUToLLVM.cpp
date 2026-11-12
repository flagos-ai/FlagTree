#include "Allocation.h"
#include "Passes.h"
#include "PatternPopulation.h"
#include "TargetInfo.h"
#include "intellifusion/lib/TritonGPNPUTransforms/ClusterBarrierInsertion.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace Intellifusion {

static bool replaceInlineAsmMnemonic(std::string &asmString, StringRef from,
                                     StringRef to) {
  size_t pos = asmString.find(from.str());
  if (pos == std::string::npos)
    return false;
  asmString.replace(pos, from.size(), to.str());
  return true;
}

static void adaptScaledFp4MMAEncoding(ModuleOp mod) {
  mod.walk([](LLVM::InlineAsmOp op) {
    std::string asmString = op.getAsmString().str();
    bool changed =
        replaceInlineAsmMnemonic(asmString, "mxf4nvf4.block_scale.block16",
                                 "mxf4nvf4.block_scale.scale_vec::4X");
    changed |= replaceInlineAsmMnemonic(asmString, "mxf4.block_scale.block32",
                                        "mxf4.block_scale.scale_vec::2X");
    if (changed)
      op.setAsmString(asmString);
  });
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addIllegalOp<triton::FuncOp>();
    markUnknownOpDynamicallyLegal(
        [](Operation *op) { return !isa<triton::FuncOp>(op); });
  }
};

// Main Pass
struct ConvertTritonGPNPUToLLVMPass
    : public PassWrapper<ConvertTritonGPNPUToLLVMPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonGPNPUToLLVMPass);

  ConvertTritonGPNPUToLLVMPass() = default;
  explicit ConvertTritonGPNPUToLLVMPass(int nWarps) { numWarps = nWarps; }

  ConvertTritonGPNPUToLLVMPass(const ConvertTritonGPNPUToLLVMPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect,
                    triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    triton::nvgpu::NVGPUDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, ::mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    // Determine warp/thread topology.
    int threadsPerWarp = 32;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
      threadsPerWarp = attr.getInt();
    int totalWarps = numWarps;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      totalWarps = attr.getInt();

    Intellifusion::TargetInfo targetInfo(threadsPerWarp, totalWarps);

    ModuleAllocation allocation(
        mod,
        Intellifusion::getGPNPUAllocationAnalysisScratchSizeFn(targetInfo));
    // GPNPU lowers the inserted cluster barriers directly to NVVM operations.
    runGPNPUClusterBarrierInsertion(allocation);
    if (failed(runGPNPUCrossCTAMBarrierInitSyncInsertion(allocation)))
      return signalPassFailure();
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    bool requiresCluster = triton::nvidia_gpu::getModuleTwoCTAs(mod);
    mod.walk([&](Operation *op) {
      if (isa<triton::nvidia_gpu::ClusterArriveOp,
              triton::nvidia_gpu::ClusterWaitOp,
              triton::nvidia_gpu::ClusterBarrierOp>(op))
        requiresCluster = true;
    });
    mod->setAttr("ttg.requires-cluster", BoolAttr::get(ctx, requiresCluster));

    bool hasGlobalScratchAlloc = false;
    mod.walk([&](triton::gpu::GlobalScratchAllocOp) {
      hasGlobalScratchAlloc = true;
    });
    if (hasGlobalScratchAlloc)
      mlir::triton::gpu::runGlobalScratchMemoryAllocation(mod);

    // Create global_smem sized to accommodate AllocateSharedMemory allocations.
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(mod.getBody());
    auto i8Type = builder.getIntegerType(8);
    auto arrayType = LLVM::LLVMArrayType::get(i8Type, 0);
    LLVM::GlobalOp::create(builder, mod.getLoc(), arrayType,
                           /*isConstant=*/false, LLVM::Linkage::External,
                           "global_smem", Attribute(), /*alignment=*/16,
                           /*addrSpace=*/3);

    // Setup conversion with the shared TritonGPUToLLVMTypeConverter
    mlir::LowerToLLVMOptions option(ctx);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(ctx, option, targetInfo);

    mod.walk([&](triton::FuncOp funcOp) {
      if (!triton::isKernel(funcOp))
        return;
      for (unsigned index = 0; index < funcOp.getNumArguments(); ++index) {
        if (isa<triton::TensorDescInterface>(
                funcOp.getArgument(index).getType()))
          funcOp.setArgAttr(index, "tt.nv_tma_desc",
                            IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    });

    TritonLLVMFunctionConversionTarget funcTarget(*ctx);
    RewritePatternSet funcPatterns(ctx);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    bool hasUnsupportedF16DivF = false;
    mod.walk([&](arith::DivFOp op) {
      if (getElementTypeOrSelf(op.getType()).isF16()) {
        op.emitError("unsupported f16 arith.divf lowering for GPNPU");
        hasUnsupportedF16DivF = true;
      }
    });
    if (hasUnsupportedF16DivF)
      return signalPassFailure();

    // Set up conversion target
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect,
                           triton::nvgpu::NVGPUDialect>();
    target
        .addIllegalDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                           triton::nvidia_gpu::TritonNvidiaGPUDialect,
                           ::mlir::gpu::GPUDialect, arith::ArithDialect>();
    target.addDynamicallyLegalOp<triton::gpu::GlobalScratchAllocOp>(
        [](triton::gpu::GlobalScratchAllocOp op) {
          return !op->getParentOfType<LLVM::LLVMFuncOp>() ||
                 op.getThirdPartyAllocation();
        });
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Populate patterns
    RewritePatternSet patterns(ctx);

    // Axis info analysis (used by elementwise patterns for deduplication opt)
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    auto benefit = mlir::triton::patternBenefitPrioritizeOverLLVMConversions;

    // --- GPNPU-specific patterns (custom lowering) ---
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    populateGPNPUArithToLLVMPatterns(patterns, typeConverter);
    populateGPNPUMemoryToLLVMPatterns(patterns, typeConverter, targetInfo,
                                      axisInfoAnalysis, benefit);

    // --- Standard MLIR conversion patterns ---
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    populateGPNPUTMAToLLVMPatterns(typeConverter, targetInfo, patterns,
                                   benefit);
    populateGPNPUTMABarrierOpToLLVMPatterns(typeConverter, patterns, benefit,
                                            targetInfo);
    populateGPNPUClusterOpToLLVMPatterns(typeConverter, patterns, benefit,
                                         targetInfo);
    populateGPNPUTMALoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    // --- Shared patterns from lib/Conversion/TritonGPUToLLVM ---
    populateGPNPUElementwiseToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    populateGPNPUMathToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                    targetInfo, benefit);
    populateGPNPUDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);

    // Match upstream scratch allocation for all shared-memory conversions,
    // using GPNPU memory hooks and inline PTX matrix stores.
    populateGPNPUConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
    // --- TCGen5 matmul bridge patterns ---
    populateGPNPUTensorMemoryOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateGPNPUTensorMemorySubviewOpToLLVMPatterns(typeConverter, patterns,
                                                     benefit);
    populateGPNPUTCGen5MMAOpToLLVMPatterns(typeConverter, patterns, benefit,
                                           targetInfo);
    // DecomposeScaledBlocked lowers unsupported FP4 tt.dot_scaled operations
    // through ttg.fp4_to_fp before an ordinary MMAv5 or GPNPU FMA dot.
    populateGPNPUFp4ToFpToLLVMPatterns(typeConverter, patterns, benefit);

    // Reduce/scan + ConvertLayout (reduce synthesizes ttg.convert_layout, so
    // both are required).
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);

    // gpu.barrier -> nvvm.barrier0 (emitted by upstream BarrierOpConversion
    // when lowering ttg.barrier inserted by Membar analysis).
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    // --- GPNPU-specific SPMD patterns (gpu::ThreadIdOp, WarpIdOp) ---
    populateGPNPUSPMDOpToLLVMPatterns(typeConverter, patterns, benefit);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      llvm::errs() << "[TritonGPNPUToLLVM] applyPartialConversion failed\n";
      signalPassFailure();
      return;
    }
    // Match the upstream post-conversion loop metadata fixup so attributes
    // carried through CF lowering become LLVM branch properties before
    // translation.
    fixUpLoopAnnotation(mod);
    // Triton 3.8's shared MMAv5 lowering emits Blackwell/Rubin FP4 mnemonic
    // spellings. GPNPU uses the same block-scaled layouts and K=64 descriptor,
    // but IFWA expects the equivalent scale-vector UMMA mnemonics.
    adaptScaledFp4MMAEncoding(mod);
  }

  StringRef getArgument() const override { return "convert-gpnpu-to-llvm"; }

  StringRef getDescription() const override { return "Convert GPNPU to LLVM"; }

private:
  Option<int> numWarps{*this, "num-warps",
                       llvm::cl::desc("number of GPNPU warps"),
                       llvm::cl::init(4)};
};

std::unique_ptr<Pass> createConvertTritonGPNPUToLLVM(int numWarps) {

  return std::make_unique<ConvertTritonGPNPUToLLVMPass>(numWarps);
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
