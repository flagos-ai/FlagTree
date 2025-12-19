#include "triton/Conversion/FlagTreeToLLVM/PackOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace fl = mlir::triton::flagtree;

struct PackOpConversion : public ConvertOpToLLVMPattern<fl::PackOp> {
  PackOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

PackOpConversion::PackOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult
PackOpConversion::matchAndRewrite(fl::PackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (ttg::MemDescType memdesc =
          dyn_cast<ttg::MemDescType>(op.getOutput().getType())) {
    LLVM::LLVMStructType llvmStructType =
        cast<LLVM::LLVMStructType>(typeConverter->convertType(memdesc));
    LLVM::ExtractValueOp basePtr = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getInput(), SmallVector<int64_t>{0});
    Value llvmStruct =
        rewriter.create<LLVM::PoisonOp>(op.getLoc(), llvmStructType);
    llvmStruct = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), llvmStructType, llvmStruct, basePtr,
        SmallVector<int64_t>{0});
    for (int64_t i = 1; i < llvmStructType.getBody().size(); ++i) {
      auto zeroOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getIntegerType(32), 0);
      llvmStruct = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), llvmStructType, llvmStruct, zeroOp,
          SmallVector<int64_t>{i});
    }
    rewriter.replaceAllOpUsesWith(op, llvmStruct);
    return success();
  } else {
    return failure();
  }
}

void fl::populatePackOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<PackOpConversion>(typeConverter, benefit);
}
