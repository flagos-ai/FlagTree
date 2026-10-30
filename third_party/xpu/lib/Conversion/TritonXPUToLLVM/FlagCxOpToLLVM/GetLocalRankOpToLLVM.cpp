#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include <string>

using namespace mlir;

namespace {

template <typename Op>
struct FlagCXCommQueryConversion : public ConvertOpToLLVMPattern<Op> {
  FlagCXCommQueryConversion(const LLVMTypeConverter &converter,
                            StringRef symbol, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Op>(converter, benefit), symbol(symbol.str()) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto resultTy = rewriter.getI32Type();
    auto fnTy = LLVM::LLVMFunctionType::get(resultTy, {ptrTy});
    auto mod = op->template getParentOfType<ModuleOp>();
    auto func = mod.template lookupSymbol<LLVM::LLVMFuncOp>(symbol);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, symbol, fnTy);
    } else if (func.getFunctionType() != fnTy) {
      return rewriter.notifyMatchFailure(
          op, "FlagCX device function has an incompatible signature");
    }
    Value comm =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, adaptor.getComm());
    auto result = rewriter.create<LLVM::CallOp>(loc, func, ValueRange{comm});
    rewriter.replaceOp(op, result.getResult());
    return success();
  }

private:
  std::string symbol;
};

} // namespace

void mlir::triton::xpu::populateGetLocalRankOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetLocalRankOp>>(
      typeConverter, "flagcxDevCommGetIntraRank", benefit);
}

void mlir::triton::xpu::populateGetNumPesOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetNumPesOp>>(
      typeConverter, "flagcxDevCommGetIntraSize", benefit);
}
