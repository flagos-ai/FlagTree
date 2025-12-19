#include "triton/Conversion/FlagTreeToLLVM/ExtractOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/LogicalResult.h"
#include <numeric>

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace fl = mlir::triton::flagtree;

struct ExtractAllocatedPtrOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractAllocatedPtrOp> {
  ExtractAllocatedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractAlignedPtrOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractAlignedPtrOp> {
  ExtractAlignedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractAlignedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractOffsetOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractOffsetOp> {
  ExtractOffsetOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractSizesOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractSizesOp> {
  ExtractSizesOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractSizesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractStridesOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractStridesOp> {
  ExtractStridesOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractStridesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
}; // namespace

ExtractAllocatedPtrOpConversion::ExtractAllocatedPtrOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractAllocatedPtrOpConversion::matchAndRewrite(
    fl::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  LLVM::ExtractValueOp newOp = rewriter.create<LLVM::ExtractValueOp>(
      op.getLoc(), adaptor.getInput(), SmallVector<int64_t>{0});
  rewriter.replaceAllUsesWith(op, newOp);
  return success();
}

ExtractAlignedPtrOpConversion::ExtractAlignedPtrOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractAlignedPtrOpConversion::matchAndRewrite(
    fl::ExtractAlignedPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  LLVM::ExtractValueOp newOp = rewriter.create<LLVM::ExtractValueOp>(
      op.getLoc(), adaptor.getInput(), SmallVector<int64_t>{0});
  rewriter.replaceAllUsesWith(op, newOp);
  return success();
}

ExtractOffsetOpConversion::ExtractOffsetOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractOffsetOpConversion::matchAndRewrite(
    fl::ExtractOffsetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                 rewriter.getI64IntegerAttr(0));
  return success();
}

ExtractSizesOpConversion::ExtractSizesOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractSizesOpConversion::matchAndRewrite(
    fl::ExtractSizesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (ttg::MemDescType memdesc =
          dyn_cast<ttg::MemDescType>(op.getInput().getType())) {
    SmallVector<Value> sizes;
    for (int64_t size : memdesc.getShape()) {
      auto newOp = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(size));
      sizes.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, sizes);
    return success();
  } else {
    return failure();
  }
}

ExtractStridesOpConversion::ExtractStridesOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractStridesOpConversion::matchAndRewrite(
    fl::ExtractStridesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (ttg::MemDescType memdesc =
          dyn_cast<ttg::MemDescType>(op.getInput().getType())) {
    SmallVector<Value> strides;
    ArrayRef<int64_t> shape = memdesc.getShape();
    int64_t numel = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int64_t>());
    for (int64_t size : memdesc.getShape()) {
      numel /= size;
      auto newOp = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(numel));
      strides.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, strides);
    return success();
  } else {
    return failure();
  }
}

void fl::populateExtractOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<ExtractAllocatedPtrOpConversion, ExtractAlignedPtrOpConversion,
               ExtractOffsetOpConversion, ExtractSizesOpConversion,
               ExtractStridesOpConversion>(typeConverter, benefit);
}
