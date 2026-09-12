/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Dialect/PPUGPU/IR/Dialect.h"
#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "TritonPPUGPUToLLVM/Passes.h"
#include "TritonPPUGPUToLLVM/Utility.h"
#include "Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#ifdef __TLE__
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#endif
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#ifdef __TLE__
#include "tle/dialect/include/Analysis/AxisInfoExt.h"
#include "tle/dialect/include/Conversion/TleToLLVM/DSLRegionOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/ExclusiveCumsumOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/ExtractOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/PackOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#endif
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVMPPU
#include "TritonPPUGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::ppu;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<mlir::triton::ppugpu::PPUGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::ppu_gpu::TritonPPUGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
#ifdef __TLE__
    addIllegalDialect<tle::TleDialect>();
#endif
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

#ifdef __TLE__
// Partial conversion target for the TLE lowering pre-pass on PPU.
// PPU only ships a subset of TLE: local_pointers, extract*, pack, extract_tile,
// insert_tile, dsl_region. WGMMA/TMA/distributed_barrier/exclusive_cumsum/
// remote_pointers are NVIDIA-only at this stage — they are left out of the
// pattern set and will produce a legalization failure if a kernel emits them,
// which is the intended behavior until those features are ported.
class TleLLVMConversionTarget : public ConversionTarget {
public:
  explicit TleLLVMConversionTarget(MLIRContext &ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, math::MathDialect,
                    NVVM::NVVMDialect, mlir::gpu::GPUDialect>();
    addIllegalDialect<tle::TleDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addDynamicallyLegalOp<tle::DSLRegionOp, tle::YieldOp>(
        [&](Operation *op) -> bool {
          bool hasLegalRegions = true;
          for (auto &region : op->getRegions()) {
            hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
          }
          return hasLegalRegions && typeConverter.isLegal(op);
        });
    // Let non-TLE ops survive this partial conversion.
    markUnknownOpDynamicallyLegal([](Operation *) -> bool { return true; });
  }
};
#endif

static void MarkChainedDot(ModuleOp mod) {
  SmallVector<triton::gpu::ConvertLayoutOp> cvtACandidates;

  mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
    auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
    auto dstType = cast<RankedTensorType>(cvtOp.getType());
    auto dstDotOpEnc =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    auto srcPPUMma =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(srcType.getEncoding());
    if (srcPPUMma && dstDotOpEnc &&
        mlir::LLVM::PPU::matchMmaV1AndDotOperandLayout(srcType, dstType)) {
      // only for PPU vecSize = 1, FP32_FP16_FP16_FP32
      if (srcPPUMma.getVecSize() == 1) {
        cvtACandidates.push_back(cvtOp);
      }
    }
  });

  for (unsigned i = 0; i < cvtACandidates.size(); i++) {
    auto cvtOp = cvtACandidates[i];
    auto dstType = cast<RankedTensorType>(cvtOp.getType());
    auto dstDotOpEnc =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    // handle the one use condition
    Value dotOpndA = cvtOp.getResult();
    if (dotOpndA.hasOneUse()) {
      auto useOp = dotOpndA.use_begin()->getOwner();
      if (!isa<DotOp>(useOp))
        continue;
      auto chainedDot = cast<DotOp>(useOp);
      Value dotOpndB = chainedDot.getB();
      // both A&B have one use
      if (dotOpndB.hasOneUse()) {
        auto BOp = dotOpndB.getDefiningOp();
        if (!isa<mlir::triton::gpu::LocalLoadOp>(BOp))
          continue;
        auto cvtBOp = cast<mlir::triton::gpu::LocalLoadOp>(BOp);
        Value src = cvtBOp.getSrc();
        auto descTy = cast<mlir::triton::gpu::MemDescType>(src.getType());
        auto encoding = descTy.getEncoding();
        if (auto srcSharedLayout =
                dyn_cast<triton::gpu::PPUAIUSharedEncodingAttr>(encoding)) {
          // TODO: temporarily disable DotOperand1 ChainedDot Optimiation for
          // AIU load if (srcSharedLayout.getOrder()[0] == 0)
          continue;
        } else if (auto srcSharedLayout =
                       dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
                           encoding)) {
          if (srcSharedLayout.getOrder()[0] == 0)
            continue;
        } else {
          continue;
        }

        // needTrans = kOrder != order[0];
        // matrixB load opt only applied on untransposed B.
        auto dstBType = cast<RankedTensorType>(cvtBOp.getType());

        // workaround for datatypes like e5m3
        // The condition is extracted from SharedToDotOperandMMAPPUv1.cpp
        const int elemBytes = descTy.getElementTypeBitWidth() / 8;
        const int vecWidth = 4 / elemBytes;
        auto dstBEnc = cast<triton::gpu::DotOperandEncodingAttr>(
            cvtBOp.getType().getEncoding());
        int kWidth = dstBEnc.getKWidth();
        if (kWidth != vecWidth)
          continue;
        // records both A&B, which will go to the special handling in pair.
        // chainedCvtsA.insert(cvtOp);
        // chainedCvtsB.insert(cvtBOp);
        auto ctx = dstDotOpEnc.getContext();
        auto dotParentEnc = dstDotOpEnc.getParent();
        auto chainedEncA = triton::gpu::DotOperandEncodingAttr::get(
            ctx, dstDotOpEnc.getOpIdx(), dotParentEnc, dstDotOpEnc.getKWidth(),
            true);
        auto newAType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(), chainedEncA);
        // auto dotBParentEnc = dstBEnc.getParent();
        auto chainedEncB = triton::gpu::DotOperandEncodingAttr::get(
            ctx, dstBEnc.getOpIdx(), dstBEnc.getParent(), dstBEnc.getKWidth(),
            true);
        auto newBType = RankedTensorType::get(
            dstBType.getShape(), dstBType.getElementType(), chainedEncB);

        OpBuilder builder(cvtOp);
        auto newCvtA = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), newAType, cvtOp.getSrc());
        builder.setInsertionPointAfter(cvtBOp);
        auto newCvtB = builder.create<mlir::triton::gpu::LocalLoadOp>(
            cvtBOp.getLoc(), newBType, cvtBOp.getSrc());
        cvtOp.replaceAllUsesWith(newCvtA.getResult());
        cvtBOp.replaceAllUsesWith(newCvtB.getResult());
        cvtOp.erase();
        cvtBOp.erase();
      }
    }
  }
}

struct ConvertTritonGPUToLLVMPPU
    : public triton::impl::ConvertTritonGPUToLLVMPPUBase<
          ConvertTritonGPUToLLVMPPU> {
  using ConvertTritonGPUToLLVMPPUBase::ConvertTritonGPUToLLVMPPUBase;

  ConvertTritonGPUToLLVMPPU(int32_t computeCapability)
      : ConvertTritonGPUToLLVMPPUBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, mlir::triton::ppu_gpu::getPPUAllocationAnalysisScratchSizeFn(
                 targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    MarkChainedDot(mod);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
#ifdef __TLE__
    mlir::triton::tle::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
#else
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
#endif

    int benefit = patternBenefitPrioritizeOverLLVMConversions;

#ifdef __TLE__
    {
      TleLLVMConversionTarget tleTarget(*context, typeConverter);
      RewritePatternSet tlePatterns(context);
      // local_pointers + Extract*/Pack/DSL-region/extract_tile/insert_tile —
      // the subset of TLE ops PPU supports today. Patterns for distributed
      // barriers, exclusive cumsum, WGMMA descriptor views, WGMMA fences and
      // TMA store commit groups are intentionally not registered: those ops
      // never appear on PPU and a legalization failure is the right signal if
      // a kernel does emit them.
      mlir::triton::tle::populateDSLRegionOpToLLVMPatterns(
          typeConverter, tlePatterns, benefit);
      mlir::triton::tle::populateExtractOpToLLVMPatterns(typeConverter,
                                                         tlePatterns, benefit);
      mlir::triton::tle::populatePackOpToLLVMPatterns(typeConverter,
                                                      tlePatterns, benefit);
      mlir::triton::tle::populateLocalPointersOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns, benefit);
      mlir::triton::tle::populateExtractTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo, benefit);
      mlir::triton::tle::populateInsertTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo, benefit);
      mlir::triton::tle::populateExclusiveCumsumOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns, benefit);
      if (failed(
              applyPartialConversion(mod, tleTarget, std::move(tlePatterns)))) {
        return signalPassFailure();
      }
    }
#endif

    RewritePatternSet patterns(context);
    mlir::triton::ppu::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        targetInfo, benefit);
    populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                  computeCapability,
                                  patternBenefitClampOptimizedPattern);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                      computeCapability, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::ppu::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::ppu::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::ppu::populateFp4ToFpToLLVMPatterns(
        typeConverter, patterns, computeCapability, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately to avoid breaking analysis.
    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1) {
      mod.walk([](triton::ppugpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPPUPass() {
  return std::make_unique<ConvertTritonGPUToLLVMPPU>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPPUPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVMPPU>(computeCapability);
}

} // namespace triton
} // namespace mlir
