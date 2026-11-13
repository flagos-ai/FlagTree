#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceInitOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

// XSHMEMI_LOCAL_BUF_LEN / __simd__ from the vendor's xshmem_common.h.
// flagcxDevInit takes the scratch as an argument because XTDK's __local__ is
// only an alignment attribute, so it has to come from the kernel's own frame.
constexpr int kXshmemLocalBufBytes = 4096;
constexpr int kXshmemLocalBufAlign = 64;

struct DeviceInitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::DeviceInitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::DeviceInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto commTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto bufTy = LLVM::LLVMPointerType::get(ctx);
    auto i8Ty = rewriter.getI8Type();
    auto i32Ty = rewriter.getI32Type();
    auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {commTy, bufTy});

    auto mod = op->getParentOfType<ModuleOp>();
    StringRef symbol = "flagcxDevInit";
    auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      func = rewriter.create<LLVM::LLVMFuncOp>(loc, symbol, fnTy);
    } else if (func.getFunctionType() != fnTy) {
      return rewriter.notifyMatchFailure(op,
                                         "flagcxDevInit signature mismatch");
    }

    Value size = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(kXshmemLocalBufBytes));
    Value buf = rewriter.create<LLVM::AllocaOp>(loc, bufTy, i8Ty, size,
                                                kXshmemLocalBufAlign);
    Value comm =
        rewriter.create<LLVM::IntToPtrOp>(loc, commTy, adaptor.getComm());
    rewriter.create<LLVM::CallOp>(loc, func, ValueRange{comm, buf});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceInitOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DeviceInitOpConversion>(typeConverter, benefit);
}
