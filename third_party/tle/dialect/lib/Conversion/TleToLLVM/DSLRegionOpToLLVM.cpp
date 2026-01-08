#include "tle/dialect/include/Conversion/TleToLLVM/DSLRegionOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/LogicalResult.h"

namespace {
using namespace mlir;
namespace tle = mlir::triton::tle;

struct DSLRegionOpConversion : public ConvertOpToLLVMPattern<tle::DSLRegionOp> {
  DSLRegionOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::DSLRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct YieldOpConversion : public ConvertOpToLLVMPattern<tle::YieldOp> {
  YieldOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

DSLRegionOpConversion::DSLRegionOpConversion(LLVMTypeConverter &typeConverter,
                                             PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult DSLRegionOpConversion::matchAndRewrite(
    tle::DSLRegionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto newOp = rewriter.cloneWithoutRegions<tle::DSLRegionOp>(op);
  Region &body = op.getBody();
  Region &newBody = newOp.getBody();
  rewriter.inlineRegionBefore(body, newBody, newBody.end());

  if (failed(rewriter.convertRegionTypes(&newBody, *getTypeConverter()))) {
    return rewriter.notifyMatchFailure(op, "could not convert body types");
  }
  newOp->setOperands(adaptor.getOperands());
  for (OpResult result : newOp.getResults()) {
    result.setType(getTypeConverter()->convertType(result.getType()));
  }
  rewriter.replaceOp(op, newOp->getResults());

  return success();
}

YieldOpConversion::YieldOpConversion(LLVMTypeConverter &typeConverter,
                                     PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult
YieldOpConversion::matchAndRewrite(tle::YieldOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<tle::YieldOp>(op, adaptor.getOperands());
  return success();
}

void tle::populateDSLRegionOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DSLRegionOpConversion, YieldOpConversion>(typeConverter,
                                                         benefit);
}
