// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::tle {

inline bool isFlagcxBlockBarrierCall(LLVM::CallOp call) {
  auto name = call.getCallee();
  if (!name || (*name != "flagcxIntraBarrierArriveS" &&
                *name != "flagcxIntraBarrierWaitS" &&
                *name != "flagcxIntraBarrierSyncS") ||
      call.getNumOperands() != 5)
    return false;
  auto constant = call.getOperand(1).getDefiningOp<LLVM::ConstantOp>();
  auto kind = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                       : IntegerAttr();
  return kind && kind.getInt() == 2;
}

// FlagCX is linked after warp-specialize lowering. Pass the allocated physical
// group through its C ABI rather than letting the library reconstruct a CTA.
inline bool scopeFlagcxBlockBarrierCall(LLVM::CallOp call, unsigned firstWarp,
                                        unsigned numThreads,
                                        unsigned barrierIdx) {
  if (!isFlagcxBlockBarrierCall(call))
    return false;
  std::string name = call.getCallee()->drop_back().str() + "WarpGroupS";
  OpBuilder builder(call);
  auto i32 = builder.getI32Type();
  auto integer = [&](unsigned value) -> Value {
    return LLVM::ConstantOp::create(builder, call.getLoc(), i32,
                                    builder.getI32IntegerAttr(value));
  };
  SmallVector<Value> args{call.getOperand(0), integer(firstWarp),
                          integer(numThreads / 32), integer(barrierIdx),
                          call.getOperand(2), call.getOperand(3),
                          call.getOperand(4)};
  ModuleOp module = call->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    OpBuilder declaration(module.getBodyRegion());
    SmallVector<Type> types;
    for (Value value : args)
      types.push_back(value.getType());
    auto signature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(module.getContext()), types, false);
    LLVM::LLVMFuncOp::create(declaration, call.getLoc(), name, signature);
  }
  call.setCalleeAttr(FlatSymbolRefAttr::get(module.getContext(), name));
  call.getCalleeOperandsMutable().assign(args);
  return true;
}

} // namespace mlir::triton::tle
