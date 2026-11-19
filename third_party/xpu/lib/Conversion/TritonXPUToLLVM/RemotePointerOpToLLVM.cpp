#include "PatternTritonXPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

// flagcxDevPeerAccessWriteOnly in flagcx_device_enums.h.
constexpr int32_t kPeerAccessWriteOnly = 1;

struct RemotePointerConversion
    : public ConvertOpToLLVMPattern<triton::xpu::RemotePointerOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::RemotePointerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    if (getTypeConverter()->convertType(op.getType()) != ptrTy)
      return op.emitOpError(
          "remote pointer must use XPU global address space 1");
    auto fnTy =
        LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, i64Ty, i32Ty, i32Ty});
    auto mod = op->getParentOfType<ModuleOp>();
    constexpr auto symbol = "flagcxGetIntraPointerWithAccessC";
    auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, symbol, fnTy);
    } else if (func.getFunctionType() != fnTy) {
      return rewriter.notifyMatchFailure(
          op, "FlagCX remote pointer signature mismatch");
    }
    Value memory =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, adaptor.getMemory());
    Value elementBytes = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(op.getElementBytes()));
    Value byteOffset =
        rewriter.create<LLVM::MulOp>(loc, adaptor.getOffset(), elementBytes);
    Value access = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(kPeerAccessWriteOnly));
    auto result = rewriter.create<LLVM::CallOp>(
        loc, func, ValueRange{memory, byteOffset, adaptor.getPeer(), access});
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateRemotePointerOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<RemotePointerConversion>(typeConverter, benefit);
}
