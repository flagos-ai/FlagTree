/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::tle {
using namespace mlir;

static const llvm::StringMap<StringRef> runtimeNames = {
    {"getLocalPeFunction", "flagcxDevCommGetIntraRank"},
    {"getWorldRankFunction", "flagcxDevCommGetRank"},
    {"getNumPesFunction", "flagcxDevCommGetIntraSize"},
    {"getIntraBarrierArriveSignalFunction", "flagcxIntraBarrierArriveS"},
    {"getIntraBarrierWaitSignalFunction", "flagcxIntraBarrierWaitS"},
    {"getIntraBarrierSyncSignalFunction", "flagcxIntraBarrierSyncS"},
    {"signalIncFunction", "flagcxDevSignalInc"},
    {"signalAddFunction", "flagcxDevSignalAdd"},
    {"waitSignalFunction", "flagcxDevWaitSignal"},
    {"waitShadowFunction", "flagcxDevWaitSignalMeetShadow"},
    {"waitCounterFunction", "flagcxDevWaitCounter"}};

static inline LLVM::LLVMFuncOp createFuncInstance(const char *funcName,
                                                  ModuleOp module,
                                                  ArrayRef<Type> argTypes,
                                                  Type returnType) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;
  auto funcType = LLVM::LLVMFunctionType::get(returnType, argTypes, false);

  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);

  func.setLinkage(LLVM::Linkage::External);
  return func;
}

// The frontend passes the FlagCX global memory/communication pointer as an
// integer. Convert it back to an LLVM pointer in global address space (AS1)
// before passing it to device/runtime functions.
static inline Value getFlagcxMemOrCommPtr(mlir::Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value memPtrInt) {
  auto ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
  return rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, memPtrInt);
}

LLVM::CallOp getNumPesFunCall(mlir::Location loc,
                              ConversionPatternRewriter &rewriter,
                              Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func = createFuncInstance(
      runtimeNames.lookup("getNumPesFunction").data(), module, {PtrTy}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemOrCommPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});
}

LLVM::CallOp getWorldRankFuncCall(mlir::Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func =
      createFuncInstance(runtimeNames.lookup("getWorldRankFunction").data(),
                         module, {ptrTy}, i32Ty);

  auto commDevPtr = getFlagcxMemOrCommPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{commDevPtr});
}

LLVM::CallOp getBarrierFuncCall(mlir::Location loc,
                                ConversionPatternRewriter &rewriter, Value comm,
                                size_t barrier_index, size_t coopKind,
                                size_t order, llvm::StringRef barrierType) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto funcName = "";
  if (barrierType == "arrive") {
    funcName = "getIntraBarrierArriveSignalFunction";
  } else if (barrierType == "wait") {
    funcName = "getIntraBarrierWaitSignalFunction";
  } else if (barrierType == "sync") {
    funcName = "getIntraBarrierSyncSignalFunction";
  } else {
    llvm_unreachable("Unknown barrier type");
  }

  auto func = createFuncInstance(runtimeNames.lookup(funcName).data(), module,
                                 {PtrTy, i32Ty, i32Ty, i1Ty, i32Ty}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemOrCommPtr(loc, rewriter, comm);
  auto falseVal =
      rewriter.create<LLVM::ConstantOp>(loc, i1Ty, rewriter.getBoolAttr(false));
  auto barrierIndexVal =
      rewriter.create<LLVM::ConstantOp>(loc, i32Ty, barrier_index);
  auto coopKindVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, coopKind);
  auto orderVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, order);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func),
      ValueRange{comm_dev_ptr, coopKindVal, barrierIndexVal, falseVal,
                 orderVal});
}

LLVM::CallOp getLocalPeFuncCall(mlir::Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value memPtrInt) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto func = createFuncInstance(
      runtimeNames.lookup("getLocalPeFunction").data(), module, {PtrTy}, i32Ty);

  auto comm_dev_ptr = getFlagcxMemOrCommPtr(loc, rewriter, memPtrInt);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr});
}

LLVM::CallOp getSignalFuncCall(mlir::Location loc,
                               ConversionPatternRewriter &rewriter, Value comm,
                               Value peer, Value slotId, Value value,
                               uint32_t contextId, SignalTeamKind teamKind,
                               SignalCoopKind coopKind, SignalOpKind signalOp) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto commPtr = getFlagcxMemOrCommPtr(loc, rewriter, comm);

  auto teamKindValue = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(teamKind)));
  auto coopKindValue = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(coopKind)));
  auto contextIdValue = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(contextId)));
  // flagcxDevMemoryScopeDevice (=1), see flagcx_device_enums.h
  auto scopeValue = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, 1);

  // Unified: (comm, teamKind, peer, signal, contextId, coopKind, scope)
  SmallVector<Value> args{commPtr,        teamKindValue, peer,      slotId,
                          contextIdValue, coopKindValue, scopeValue};
  SmallVector<Type> argTypes{ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty};

  StringRef runtimeName;
  switch (signalOp) {
  case SignalOpKind::INC:
    runtimeName = runtimeNames.lookup("signalIncFunction");
    break;
  case SignalOpKind::ADD:
    runtimeName = runtimeNames.lookup("signalAddFunction");
    // Unified: (comm, teamKind, peer, signal, value, contextId, coopKind,
    // scope)
    argTypes.insert(argTypes.begin() + 4, i64Ty);
    args.insert(args.begin() + 4, value);
    break;
  default:
    llvm_unreachable("unknown signal operation");
  }

  auto signalFunc =
      createFuncInstance(runtimeName.data(), module, argTypes, voidTy);
  return rewriter.create<LLVM::CallOp>(
      loc, TypeRange{}, FlatSymbolRefAttr::get(signalFunc), args);
}

LLVM::CallOp getDevNetWaitFuncCallByKind(
    mlir::Location loc, ConversionPatternRewriter &rewriter, Value comm,
    Value slot_id, SignalWaitKind wait_kind, std::optional<Value> target,
    SignalCoopKind coop_kind, uint32_t contextId) {
  auto ctx = rewriter.getContext();
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto I32Ty = IntegerType::get(ctx, 32);
  auto I64Ty = IntegerType::get(ctx, 64);
  auto VoidTy = LLVM::LLVMVoidType::get(ctx);

  auto commPtr = getFlagcxMemOrCommPtr(loc, rewriter, comm);
  auto coop_kind_val = rewriter.create<LLVM::ConstantOp>(
      loc, I32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(coop_kind)));
  auto contextIdValue = rewriter.create<LLVM::ConstantOp>(
      loc, I32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(contextId)));
  // TODO: actually use the named enum value flagcxDeviceMemoryOrderAcquire(=1)
  // if possible
  auto order = rewriter.create<LLVM::ConstantOp>(loc, I32Ty, 1);

  LLVM::ConstantOp bits;
  LLVM::LLVMFuncOp func;
  auto make_call = [&](ValueRange args) {
    return rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                         FlatSymbolRefAttr::get(func), args);
  };

  switch (wait_kind) {
  case SignalWaitKind::COUNTER:
    // Unified: (comm, counter, least, bits, contextId, coopKind, order)
    func = createFuncInstance(
        runtimeNames.lookup("waitCounterFunction").data(), module,
        {PtrTy, I32Ty, I64Ty, I32Ty, I32Ty, I32Ty, I32Ty}, VoidTy);
    bits = rewriter.create<LLVM::ConstantOp>(loc, I32Ty, 64);
    return make_call(ValueRange{commPtr, slot_id, target.value(), bits,
                                contextIdValue, coop_kind_val, order});
  case SignalWaitKind::SIGNAL:
    // Unified: (comm, signal, least, bits, contextId, coopKind, order)
    func = createFuncInstance(
        runtimeNames.lookup("waitSignalFunction").data(), module,
        {PtrTy, I32Ty, I64Ty, I32Ty, I32Ty, I32Ty, I32Ty}, VoidTy);
    bits = rewriter.create<LLVM::ConstantOp>(loc, I32Ty, 64);
    return make_call(ValueRange{commPtr, slot_id, target.value(), bits,
                                contextIdValue, coop_kind_val, order});
  case SignalWaitKind::SHADOW:
    // Unified: (comm, contextId, signal, bits, coopKind, order)
    func = createFuncInstance(
        runtimeNames.lookup("waitShadowFunction").data(), module,
        {PtrTy, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty}, VoidTy);
    bits = rewriter.create<LLVM::ConstantOp>(loc, I32Ty, 64);
    return make_call(ValueRange{commPtr, contextIdValue, slot_id, bits,
                                coop_kind_val, order});
  default:
    llvm_unreachable("unknown wait kind");
  }
}

} // namespace mlir::triton::tle
