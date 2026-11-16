#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceSignalOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

// The signal aggregate is a sum of 64-bit words, so the narrower compare has no
// meaning here; FlagCX's own callers pass 64 as well.
constexpr int32_t kSignalCompareBits = 64;

// Declare a FlagCX device symbol on first use. A signature mismatch means this
// lowering and libflagcx_device.bc disagree on the ABI, so it must fail loudly.
static FailureOr<LLVM::LLVMFuncOp>
flagcxDecl(ConversionPatternRewriter &rewriter, Operation *op, StringRef symbol,
           LLVM::LLVMFunctionType fnTy) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    func = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), symbol, fnTy);
  } else if (func.getFunctionType() != fnTy) {
    return rewriter.notifyMatchFailure(op, "FlagCX signal signature mismatch");
  }
  return func;
}

struct SignalOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    bool isAdd = op.getSignalKindAttr().getValue() == "add";
    // Inc(comm, teamKind, peer, signal, contextId, coopKind, scope); Add
    // inserts `value` after signal.
    SmallVector<Type> params{ptrTy, i32Ty, i32Ty, i32Ty};
    if (isAdd)
      params.push_back(i64Ty);
    params.append({i32Ty, i32Ty, i32Ty});
    auto func = flagcxDecl(rewriter, op,
                           isAdd ? "flagcxDevSignalAdd" : "flagcxDevSignalInc",
                           LLVM::LLVMFunctionType::get(voidTy, params));
    if (failed(func))
      return failure();

    auto constant = [&](Type ty, int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, ty, rewriter.getIntegerAttr(ty, value));
    };
    SmallVector<Value> args{
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, adaptor.getComm()),
        constant(i32Ty, op.getTeamKindAttr().getInt()), adaptor.getPeer(),
        adaptor.getSlot()};
    if (isAdd)
      args.push_back(adaptor.getAddend());
    args.append({constant(i32Ty, op.getContextIdAttr().getInt()),
                 constant(i32Ty, op.getCoopKindAttr().getInt()),
                 constant(i32Ty, op.getScopeAttr().getInt())});
    rewriter.create<LLVM::CallOp>(loc, *func, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct SignalWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);

    StringRef kind = op.getWaitKindAttr().getValue();
    StringRef symbol = llvm::StringSwitch<StringRef>(kind)
                           .Case("signal", "flagcxDevWaitSignal")
                           .Case("counter", "flagcxDevWaitCounter")
                           .Case("shadow", "flagcxDevWaitSignalMeetShadow")
                           .Default("");
    if (symbol.empty())
      return op.emitOpError("unsupported signal wait kind");

    auto constant = [&](Type ty, int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, ty, rewriter.getIntegerAttr(ty, value));
    };
    Value comm =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, adaptor.getComm());
    Value bits = constant(i32Ty, kSignalCompareBits);
    Value contextId = constant(i32Ty, op.getContextIdAttr().getInt());
    Value coopKind = constant(i32Ty, op.getCoopKindAttr().getInt());
    Value order = constant(i32Ty, op.getOrderAttr().getInt());

    SmallVector<Type> params;
    SmallVector<Value> args;
    if (kind == "shadow") {
      // MeetShadow takes the slot after the context and has no `least`.
      params = {ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty};
      args = {comm, contextId, adaptor.getSlot(), bits, coopKind, order};
    } else {
      // WaitSignal / WaitCounter(comm, slot, least, bits, ctx, coop, order)
      params = {ptrTy, i32Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty};
      args = {comm, adaptor.getSlot(), adaptor.getLeast(),
              bits, contextId,         coopKind,
              order};
    }
    auto func = flagcxDecl(rewriter, op, symbol,
                           LLVM::LLVMFunctionType::get(voidTy, params));
    if (failed(func))
      return failure();
    rewriter.create<LLVM::CallOp>(loc, *func, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceSignalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SignalOpConversion, SignalWaitOpConversion>(typeConverter,
                                                           benefit);
}
