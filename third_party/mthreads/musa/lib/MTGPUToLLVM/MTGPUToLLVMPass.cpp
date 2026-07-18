#include "MTGPUToLLVM/MTGPUToLLVMPass.h"

#include "Dialect/MTGPU/IR/Dialect.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "../TritonMUSAGPUToLLVM/DotOpToLLVM/DotOpToLLVM.h"

#include <optional>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTMTGPUTOLLVM
#include "musa/include/MTGPUToLLVM/Passes.h.inc"

namespace mtgpu {

namespace {

struct SqmmaOpConversion
    : public ConvertOpToLLVMPattern<triton::mtgpu::SqmmaOp> {
  using ConvertOpToLLVMPattern<triton::mtgpu::SqmmaOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mtgpu::SqmmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value threadId = getThreadId(rewriter, op.getLoc());
    if (failed(mlir::triton::MUSA::convertSQMMADot(
            op, adaptor, this->getTypeConverter(), rewriter, threadId))) {
      return op.emitError("MUSA SQMMA: mtgpu lowering failed");
    }
    return success();
  }
};

struct PackSqmmaAccumulatorOpConversion
    : public ConvertOpToLLVMPattern<triton::mtgpu::PackSqmmaAccumulatorOp> {
  using ConvertOpToLLVMPattern<
      triton::mtgpu::PackSqmmaAccumulatorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mtgpu::PackSqmmaAccumulatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorTy = cast<RankedTensorType>(op.getInput().getType());
    Value packed = mlir::LLVM::MUSA::packSqmmaAccumulatorCarrierFromTensor(
        op.getLoc(), adaptor.getInput(), tensorTy, this->getTypeConverter(),
        rewriter);
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct UnpackSqmmaAccumulatorOpConversion
    : public ConvertOpToLLVMPattern<triton::mtgpu::UnpackSqmmaAccumulatorOp> {
  using ConvertOpToLLVMPattern<
      triton::mtgpu::UnpackSqmmaAccumulatorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mtgpu::UnpackSqmmaAccumulatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorTy = cast<RankedTensorType>(op.getOutput().getType());
    Value unpacked = mlir::LLVM::MUSA::unpackSqmmaAccumulatorCarrierToTensor(
        op.getLoc(), adaptor.getCarrier(), tensorTy, this->getTypeConverter(),
        rewriter);
    rewriter.replaceOp(op, unpacked);
    return success();
  }
};

struct SqmmaWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::mtgpu::SqmmaWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::mtgpu::SqmmaWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mtgpu::SqmmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                    "llvm.musa.sqmma.wait", TypeRange{}, {});
    rewriter.replaceOp(op, adaptor.getInputs());
    return success();
  }
};

} // namespace

void populateMTGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit) {
  patterns.add<SqmmaOpConversion, PackSqmmaAccumulatorOpConversion,
               UnpackSqmmaAccumulatorOpConversion, SqmmaWaitOpConversion>(
      typeConverter, benefit);
}

} // namespace mtgpu

namespace {

class ConvertMTGPUToLLVM
    : public impl::ConvertMTGPUToLLVMBase<ConvertMTGPUToLLVM> {
public:
  using impl::ConvertMTGPUToLLVMBase<
      ConvertMTGPUToLLVM>::ConvertMTGPUToLLVMBase;

  ConvertMTGPUToLLVM() = default;
  ConvertMTGPUToLLVM(int32_t computeCapability)
      : impl::ConvertMTGPUToLLVMBase<ConvertMTGPUToLLVM>({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::triton::MUSA::TargetInfo targetInfo(computeCapability);

    LowerToLLVMOptions options(context);
    options.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, options, targetInfo);
    typeConverter.addConversion(
        [&](triton::mtgpu::SqmmaAccumulatorType type) -> std::optional<Type> {
          auto info = LLVM::MUSA::getSqmmaAccumulatorCarrierInfo(type);
          if (failed(info))
            return std::nullopt;
          return info->carrierType;
        });

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addIllegalDialect<triton::mtgpu::MTGPUDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    mtgpu::populateMTGPUToLLVMPatterns(typeConverter, patterns,
                                       PatternBenefit(1));

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertMTGPUToLLVMPass() {
  return std::make_unique<ConvertMTGPUToLLVM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMTGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertMTGPUToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
