#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef symbol =
        llvm::StringSwitch<StringRef>(op.getKindAttr().getValue())
            .Case("sync", "flagcxIntraBarrierSyncS")
            .Case("arrive", "flagcxIntraBarrierArriveS")
            .Case("wait", "flagcxIntraBarrierWaitS")
            .Default("");
    if (symbol.empty())
      return op.emitOpError("unsupported device barrier kind");

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto i32Ty = rewriter.getI32Type();
    auto i1Ty = rewriter.getI1Type();
    auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {ptrTy, i32Ty, i32Ty, i1Ty, i32Ty});
    auto mod = op->getParentOfType<ModuleOp>();
    auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, symbol, fnTy);
    } else if (func.getFunctionType() != fnTy) {
      return rewriter.notifyMatchFailure(op,
                                         "FlagCX barrier signature mismatch");
    }

    auto constant = [&](Type ty, int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, ty, rewriter.getIntegerAttr(ty, value));
    };
    Value comm =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, adaptor.getComm());
    Value coopKind = constant(i32Ty, op.getCoopKindAttr().getInt());
    Value barrierIndex = constant(i32Ty, op.getBarrierIndexAttr().getInt());
    Value multimem = constant(i1Ty, 0);
    Value order = constant(i32Ty, op.getOrderAttr().getInt());
    rewriter.create<LLVM::CallOp>(
        loc, func, ValueRange{comm, coopKind, barrierIndex, multimem, order});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
