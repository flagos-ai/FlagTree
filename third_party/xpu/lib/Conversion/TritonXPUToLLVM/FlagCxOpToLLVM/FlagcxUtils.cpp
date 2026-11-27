#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

namespace mlir::triton::xpu {

LLVM::LLVMPointerType getFlagcxGlobalPtrType(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context, /*addressSpace=*/1);
}

namespace {

const llvm::StringMap<StringRef> runtimeNames = {
    {"init", "flagcxDevInit"},
    {"getLocalPe", "flagcxDevCommGetIntraRank"},
    {"getNumPes", "flagcxDevCommGetIntraSize"},
    {"getIntraPointer", "flagcxGetIntraPointerC"},
    {"barrier.arrive", "flagcxIntraBarrierArriveS"},
    {"barrier.wait", "flagcxIntraBarrierWaitS"},
    {"barrier.sync", "flagcxIntraBarrierSyncS"},
    {"signal.inc", "flagcxDevSignalInc"},
    {"signal.add", "flagcxDevSignalAdd"},
    {"wait.signal", "flagcxDevWaitSignal"},
    {"wait.counter", "flagcxDevWaitCounter"},
    {"wait.shadow", "flagcxDevWaitSignalMeetShadow"},
};

// All FlagCX device entry points take an AS1 communicator or memory pointer
// first. The frontend represents that handle as an integer.
FailureOr<LLVM::CallOp>
createFlagcxDeviceCall(Operation *op, ConversionPatternRewriter &rewriter,
                       StringRef symbol, Type result, ArrayRef<Type> argTypes,
                       Value handle, ValueRange args) {
  auto ptrTy = getFlagcxGlobalPtrType(rewriter.getContext());
  SmallVector<Type> params{ptrTy};
  params.append(argTypes.begin(), argTypes.end());
  auto type = LLVM::LLVMFunctionType::get(result, params);
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

  Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, handle);
  SmallVector<Value> operands{ptr};
  operands.append(args.begin(), args.end());
  return rewriter.create<LLVM::CallOp>(loc, func, operands);
}

Value getI32Constant(Operation *op, ConversionPatternRewriter &rewriter,
                     int32_t value) {
  return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(value));
}

} // namespace

FailureOr<LLVM::CallOp>
getDeviceInitFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                      Value comm, Value scratch) {
  auto *context = rewriter.getContext();
  return createFlagcxDeviceCall(op, rewriter, runtimeNames.lookup("init"),
                                LLVM::LLVMVoidType::get(context),
                                {LLVM::LLVMPointerType::get(context)}, comm,
                                {scratch});
}

FailureOr<LLVM::CallOp>
getLocalPeFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                   Value comm) {
  return createFlagcxDeviceCall(op, rewriter, runtimeNames.lookup("getLocalPe"),
                                rewriter.getI32Type(), {}, comm, {});
}

FailureOr<LLVM::CallOp>
getNumPesFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                  Value comm) {
  return createFlagcxDeviceCall(op, rewriter, runtimeNames.lookup("getNumPes"),
                                rewriter.getI32Type(), {}, comm, {});
}

FailureOr<LLVM::CallOp>
getIntraPointerFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                        Value memory, Value byteOffset, Value peer) {
  return createFlagcxDeviceCall(
      op, rewriter, runtimeNames.lookup("getIntraPointer"),
      getFlagcxGlobalPtrType(rewriter.getContext()),
      {rewriter.getI64Type(), rewriter.getI32Type()}, memory, {byteOffset, peer});
}

FailureOr<LLVM::CallOp>
getIntraBarrierFuncCall(Operation *op, ConversionPatternRewriter &rewriter,
                        Value comm, int32_t coopKind, int32_t barrierIndex,
                        int32_t order, StringRef barrierKind) {
  StringRef symbol = runtimeNames.lookup(("barrier." + barrierKind).str());
  if (symbol.empty()) {
    op->emitOpError("unsupported device barrier kind");
    return failure();
  }
  auto i32 = rewriter.getI32Type();
  auto i1 = rewriter.getI1Type();
  Value coop = getI32Constant(op, rewriter, coopKind);
  Value index = getI32Constant(op, rewriter, barrierIndex);
  Value multimem = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), i1, rewriter.getIntegerAttr(i1, 0));
  Value memoryOrder = getI32Constant(op, rewriter, order);
  return createFlagcxDeviceCall(
      op, rewriter, symbol, LLVM::LLVMVoidType::get(rewriter.getContext()),
      {i32, i32, i1, i32}, comm, {coop, index, multimem, memoryOrder});
}

FailureOr<LLVM::CallOp>
getSignalFuncCall(Operation *op, ConversionPatternRewriter &rewriter, Value comm,
                  int32_t teamKind, Value peer, Value slot, Value addend,
                  int32_t contextId, int32_t coopKind, int32_t scope) {
  auto i32 = rewriter.getI32Type();
  SmallVector<Type> params{i32, i32, i32};
  SmallVector<Value> args{getI32Constant(op, rewriter, teamKind), peer, slot};
  if (addend) {
    params.push_back(rewriter.getI64Type());
    args.push_back(addend);
  }
  params.append({i32, i32, i32});
  args.append({getI32Constant(op, rewriter, contextId),
               getI32Constant(op, rewriter, coopKind),
               getI32Constant(op, rewriter, scope)});
  StringRef symbol = runtimeNames.lookup(addend ? "signal.add" : "signal.inc");
  return createFlagcxDeviceCall(op, rewriter, symbol,
                                LLVM::LLVMVoidType::get(rewriter.getContext()),
                                params, comm, args);
}

FailureOr<LLVM::CallOp>
getDevNetWaitFuncCallByKind(Operation *op, ConversionPatternRewriter &rewriter,
                            Value comm, Value slot, Value least,
                            int32_t contextId, int32_t coopKind, int32_t order,
                            StringRef waitKind) {
  StringRef symbol = runtimeNames.lookup(("wait." + waitKind).str());
  if (symbol.empty()) {
    op->emitOpError("unsupported signal wait kind");
    return failure();
  }
  auto i32 = rewriter.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  // XSHMEM signal and counter aggregates use 64-bit words.
  Value bits = getI32Constant(op, rewriter, 64);
  Value context = getI32Constant(op, rewriter, contextId);
  Value coop = getI32Constant(op, rewriter, coopKind);
  Value memoryOrder = getI32Constant(op, rewriter, order);
  if (waitKind == "shadow")
    return createFlagcxDeviceCall(op, rewriter, symbol, voidTy,
                                  {i32, i32, i32, i32, i32}, comm,
                                  {context, slot, bits, coop, memoryOrder});
  return createFlagcxDeviceCall(
      op, rewriter, symbol, voidTy,
      {i32, rewriter.getI64Type(), i32, i32, i32, i32}, comm,
      {slot, least, bits, context, coop, memoryOrder});
}

} // namespace mlir::triton::xpu
