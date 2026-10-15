#include "Conversion/TleToLLVM/PackOpToLLVM.h"
#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
using namespace mlir::triton;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;

struct PackOpConversion : public ConvertOpToLLVMPattern<iluvatar_tle::PackOp> {
  PackOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(iluvatar_tle::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

PackOpConversion::PackOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult
PackOpConversion::matchAndRewrite(iluvatar_tle::PackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (ttg::MemDescType memdesc =
          dyn_cast<ttg::MemDescType>(op.getOutput().getType())) {
    LLVM::LLVMStructType llvmStructType =
        cast<LLVM::LLVMStructType>(typeConverter->convertType(memdesc));
    LLVM::ExtractValueOp basePtr = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getInput(), SmallVector<int64_t>{0});
    Value llvmStruct =
        rewriter.create<LLVM::PoisonOp>(op.getLoc(), llvmStructType);
    LLVM::InsertValueOp insertOp = rewriter.create<LLVM::InsertValueOp>(
        op.getLoc(), llvmStructType, llvmStruct, basePtr,
        SmallVector<int64_t>{0});
    for (int64_t i = 1; i < llvmStructType.getBody().size(); ++i) {
      LLVM::ConstantOp zeroOp = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getIntegerType(32), 0);
      insertOp = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), llvmStructType, insertOp, zeroOp,
          SmallVector<int64_t>{i});
    }
    rewriter.replaceOp(op, insertOp->getResults());
    return success();
  } else {
    return failure();
  }
}

void iluvatar_tle::populatePackOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<PackOpConversion>(typeConverter, benefit);
}
