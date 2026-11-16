// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/IR/SymbolTable.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Tools/WarpSpecializeTrace.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::tle {

// Invoked after the real WS runtime has configured entry scratch ownership.
// Helpers get an explicitly appended argument and every transitive direct call
// is updated; an arbitrary helper's former last argument is never repurposed.
inline LogicalResult lowerTypedTraceEvents(ModuleOp module) {
  SmallVector<TraceEventOp> events;
  module.walk([&](TraceEventOp event) { events.push_back(event); });
  if (events.empty())
    return success();
  const bool census = module->hasAttr("tle.census.occurrences");
  if (census && !module->hasAttr("tle.census.configured"))
    return events.front().emitOpError("census requires a configured WS profile-scratch ABI");
  if (!census && (!module->hasAttr("tle.ws_trace.capacity") ||
      !module->hasAttr("tle.ws_trace.total_num_warps") ||
      !module->hasAttr("tle.ws_trace.typed")))
    return events.front().emitOpError("typed trace requires an enabled, configured WS profile-scratch ABI");
  llvm::SetVector<Operation *> functions;
  for (TraceEventOp event : events) {
    if (census != (event.getKind() == 280 || event.getKind() == 281))
      return event.emitOpError("census and ring events cannot share a profile-scratch ABI");
    auto function = event->getParentOfType<LLVM::LLVMFuncOp>();
    if (!function)
      return event.emitOpError("typed trace must be inside an LLVM device function");
    functions.insert(function);
  }
  for (unsigned index = 0; index < functions.size(); ++index) {
    auto function = cast<LLVM::LLVMFuncOp>(functions[index]);
    auto uses = SymbolTable::getSymbolUses(function, module);
    if (!uses)
      return function.emitError("typed trace requires a closed direct-call graph");
    for (const auto &use : *uses) {
      auto call = dyn_cast<LLVM::CallOp>(use.getUser());
      if (!call || !call.getCallee())
        return function.emitError("typed trace does not support address-taken or indirect device functions");
      auto caller = call->getParentOfType<LLVM::LLVMFuncOp>();
      if (!caller)
        return call.emitError("typed trace caller must be an LLVM function");
      functions.insert(caller);
    }
  }
  DenseMap<Operation *, Value> pointers;
  DenseMap<Operation *, unsigned> originalArities;
  OpBuilder builder(module.getContext());
  for (Operation *operation : functions) {
    auto function = cast<LLVM::LLVMFuncOp>(operation);
    if (function.isExternal() || function.getFunctionType().isVarArg())
      return function.emitError("typed trace requires defined nonvariadic functions");
    if (function.getLinkage() != LLVM::Linkage::Internal) {
      auto index = function->getAttrOfType<IntegerAttr>(census ? "tle.census.profile_arg" : "tle.ws_trace.profile_arg");
      if (!index || index.getInt() < 0 || index.getInt() >= function.getNumArguments())
        return function.emitError("typed trace root has no configured profile pointer argument");
      Value pointer = function.getArgument(index.getInt());
      auto pointerType = dyn_cast<LLVM::LLVMPointerType>(pointer.getType());
      if (!pointerType || pointerType.getAddressSpace() != 1)
        return function.emitError("typed trace profile argument must be an address-space-1 pointer");
      pointers[operation] = pointer;
      continue;
    }
    unsigned index = function.getNumArguments();
    originalArities[operation] = index;
    Type pointerType = LLVM::LLVMPointerType::get(module.getContext(), 1);
    auto type = function.getFunctionType();
    SmallVector<Type> inputs(type.getParams());
    inputs.push_back(pointerType);
    function.setFunctionType(LLVM::LLVMFunctionType::get(type.getReturnType(), inputs, false));
    pointers[operation] = function.getBody().front().addArgument(pointerType, function.getLoc());
    if (auto attrs = function.getArgAttrsAttr()) {
      SmallVector<Attribute> expanded(attrs.getValue());
      expanded.push_back(builder.getDictionaryAttr({}));
      function.setArgAttrsAttr(builder.getArrayAttr(expanded));
    }
    function->setAttr("tle.ws_trace.forwarded_arg", builder.getI32IntegerAttr(index));
  }
  WalkResult calls = module.walk([&](LLVM::CallOp call) {
    if (!call.getCallee())
      return WalkResult::advance();
    auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee());
    auto found = originalArities.find(callee);
    if (found == originalArities.end())
      return WalkResult::advance();
    auto caller = call->getParentOfType<LLVM::LLVMFuncOp>();
    Value pointer = pointers.lookup(caller);
    if (!pointer || call.getCalleeOperands().size() != found->second) {
      call.emitError("typed trace pointer forwarding has an inconsistent callsite ABI");
      return WalkResult::interrupt();
    }
    SmallVector<Value> operands(call.getCalleeOperands());
    operands.push_back(pointer);
    call.getCalleeOperandsMutable().assign(operands);
    return WalkResult::advance();
  });
  if (calls.wasInterrupted())
    return failure();
  for (TraceEventOp event : events) {
    auto function = event->getParentOfType<LLVM::LLVMFuncOp>();
    TritonLLVMIRRewriter builder(event.getLoc(), module.getContext());
    builder.setInsertionPoint(event);
    if (census) {
      PublicationCensus::emit(builder, function, pointers.lookup(function),
                             event.getLayer(), event.getIteration(),
                             event.getSite(), event.getKind() == 281);
      event.erase();
      continue;
    }
    Value site = builder.i32_val(event.getSite());
    SmallVector<Value> payload{site, event.getLayer(), event.getIteration(), event.getSlot()};
    WarpSpecializeTrace::emit(builder, function,
                             static_cast<WarpSpecializeTrace::Event>(event.getKind()),
                             event.getPhase(), payload, pointers.lookup(function));
    event.erase();
  }
  return success();
}
} // namespace mlir::triton::tle
