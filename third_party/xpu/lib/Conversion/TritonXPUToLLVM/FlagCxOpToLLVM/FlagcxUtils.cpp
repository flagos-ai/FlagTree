#include "FlagcxUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::xpu {

LLVM::LLVMPointerType getFlagcxGlobalPtrType(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context, /*addressSpace=*/1);
}

namespace {

struct FlagcxDeviceSignature {
  StringRef symbol;
  LLVM::LLVMFunctionType type;
};

FlagcxDeviceSignature getFlagcxDeviceSignature(FlagcxDeviceFunction function,
                                               OpBuilder &builder) {
  auto *context = builder.getContext();
  Type ptr = getFlagcxGlobalPtrType(context);
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type voidTy = LLVM::LLVMVoidType::get(context);
  Type barrierArgs[] = {ptr, i32, i32, builder.getI1Type(), i32};
  Type waitArgs[] = {ptr, i32, i64, i32, i32, i32, i32};
  auto signature = [](StringRef symbol, Type result, ArrayRef<Type> args) {
    return FlagcxDeviceSignature{symbol,
                                 LLVM::LLVMFunctionType::get(result, args)};
  };

  switch (function) {
  case FlagcxDeviceFunction::Init:
    return signature("flagcxDevInit", voidTy,
                     {ptr, LLVM::LLVMPointerType::get(context)});
  case FlagcxDeviceFunction::GetIntraRank:
    return signature("flagcxDevCommGetIntraRank", i32, {ptr});
  case FlagcxDeviceFunction::GetIntraSize:
    return signature("flagcxDevCommGetIntraSize", i32, {ptr});
  case FlagcxDeviceFunction::GetIntraPointer:
    return signature("flagcxGetIntraPointerC", ptr, {ptr, i64, i32});
  case FlagcxDeviceFunction::IntraBarrierArrive:
    return signature("flagcxIntraBarrierArriveS", voidTy, barrierArgs);
  case FlagcxDeviceFunction::IntraBarrierWait:
    return signature("flagcxIntraBarrierWaitS", voidTy, barrierArgs);
  case FlagcxDeviceFunction::IntraBarrierSync:
    return signature("flagcxIntraBarrierSyncS", voidTy, barrierArgs);
  case FlagcxDeviceFunction::SignalInc:
    return signature("flagcxDevSignalInc", voidTy,
                     {ptr, i32, i32, i32, i32, i32, i32});
  case FlagcxDeviceFunction::SignalAdd:
    return signature("flagcxDevSignalAdd", voidTy,
                     {ptr, i32, i32, i32, i64, i32, i32, i32});
  case FlagcxDeviceFunction::WaitSignal:
    return signature("flagcxDevWaitSignal", voidTy, waitArgs);
  case FlagcxDeviceFunction::WaitCounter:
    return signature("flagcxDevWaitCounter", voidTy, waitArgs);
  case FlagcxDeviceFunction::WaitSignalMeetShadow:
    return signature("flagcxDevWaitSignalMeetShadow", voidTy,
                     {ptr, i32, i32, i32, i32, i32});
  }
  llvm_unreachable("unknown FlagCX device function");
}

} // namespace

FailureOr<LLVM::CallOp>
createFlagcxDeviceCall(Operation *op, ConversionPatternRewriter &rewriter,
                       FlagcxDeviceFunction function, Value handle,
                       ValueRange args) {
  auto [symbol, type] = getFlagcxDeviceSignature(function, rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
  auto loc = op->getLoc();
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    func = rewriter.create<LLVM::LLVMFuncOp>(loc, symbol, type);
  } else if (func.getFunctionType() != type) {
    return rewriter.notifyMatchFailure(
        op, llvm::Twine("FlagCX device signature mismatch for ") + symbol);
  }

  auto ptr = rewriter.create<LLVM::IntToPtrOp>(
      loc, getFlagcxGlobalPtrType(rewriter.getContext()), handle);
  SmallVector<Value> operands{ptr};
  operands.append(args.begin(), args.end());
  return rewriter.create<LLVM::CallOp>(loc, func, operands);
}

} // namespace mlir::triton::xpu
