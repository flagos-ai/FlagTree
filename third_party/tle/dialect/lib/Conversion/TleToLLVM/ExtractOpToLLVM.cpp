#include "tle/dialect/include/Conversion/TleToLLVM/ExtractOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/LogicalResult.h"
#include <numeric>

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

struct ExtractAllocatedPtrOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractAllocatedPtrOp> {
  ExtractAllocatedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractAlignedPtrOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractAlignedPtrOp> {
  ExtractAlignedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractAlignedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractOffsetOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractOffsetOp> {
  ExtractOffsetOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractSizesOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractSizesOp> {
  ExtractSizesOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractSizesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractStridesOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractStridesOp> {
  ExtractStridesOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractStridesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractPtrOpConversion
    : public ConvertOpToLLVMPattern<tle::ExtractPtrOp> {
  ExtractPtrOpConversion(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::ExtractPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

ExtractAllocatedPtrOpConversion::ExtractAllocatedPtrOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractAllocatedPtrOpConversion::matchAndRewrite(
    tle::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
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
    tle::ExtractAlignedPtrOp op, OpAdaptor adaptor,
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
    tle::ExtractOffsetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                 rewriter.getI64IntegerAttr(0));
  return success();
}

ExtractSizesOpConversion::ExtractSizesOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractSizesOpConversion::matchAndRewrite(
    tle::ExtractSizesOp op, OpAdaptor adaptor,
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
    tle::ExtractStridesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (ttg::MemDescType memdesc =
          dyn_cast<ttg::MemDescType>(op.getInput().getType())) {
    ArrayRef<int64_t> shape = memdesc.getShape();
    unsigned rank = memdesc.getShape().size();
    SmallVector<int64_t> strides(rank, 0);
    llvm::SmallVector<uint32_t> order = ttg::getOrder(memdesc);
    int64_t running = 1;
    for (auto &elem: order) {
      unsigned axis = elem;
      strides[axis] = running;
      running *= shape[axis];
    }
    llvm::SmallVector<Value> strideValues;
    for (unsigned i = 0; i < shape.size(); ++i) {
      strideValues.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(strides[i])));
    }
    rewriter.replaceOpWithMultiple(op, strideValues);
    return success();
  } else {
    return failure();
  }
}

ExtractPtrOpConversion::ExtractPtrOpConversion(LLVMTypeConverter &typeConverter,
                                               PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractPtrOpConversion::matchAndRewrite(
    tle::ExtractPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto input = adaptor.getInput();
  if (isa<LLVM::LLVMPointerType>(input.getType())) {
    rewriter.replaceOp(op, input);
    return success();
  } else {
    return failure();
  }
}

void tle::populateExtractOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit) {
  patterns.add<ExtractAllocatedPtrOpConversion, ExtractAlignedPtrOpConversion,
               ExtractOffsetOpConversion, ExtractSizesOpConversion,
               ExtractStridesOpConversion, ExtractPtrOpConversion>(
      typeConverter, benefit);
}
