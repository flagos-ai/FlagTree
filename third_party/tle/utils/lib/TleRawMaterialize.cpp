/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
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

#include "tle/utils/include/TleRawMaterialize.h"
#include "ir.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "tle/utils/include/Protocol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;
namespace tle = mlir::triton::tle;

namespace {
SmallVector<Value> flatten(TritonOpBuilder &builder,
                           const TypedValue<LLVM::LLVMStructType> &val) {
  LLVM::LLVMStructType llvmStructTy = val.getType();
  const size_t rank = llvmStructTy.getBody().size();
  return llvm::map_to_vector(
      llvm::seq(rank), [&builder, &val](int64_t idx) -> Value {
        return builder.create<LLVM::ExtractValueOp>(val, SmallVector{idx});
      });
}
} // namespace

namespace mlir::triton::tle::raw {

OwningOpRef<ModuleOp> parseLLVMModule(MLIRContext *context,
                                      llvm::StringRef text) {
  ParserConfig config(context);
  return parseSourceString<ModuleOp>(text, config);
}

LLVM::LLVMFuncOp findExternalLLVMFunc(ModuleOp module,
                                      std::optional<llvm::StringRef> name) {
  if (name) {
    if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(*name))
      return func;
    for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
      if (op.getSymName().contains(*name))
        return op;
    }
    return nullptr;
  }
  LLVM::LLVMFuncOp func = nullptr;
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    if (!op.empty() && op.getLinkage() != LLVM::Linkage::Internal) {
      if (func)
        return nullptr;
      func = op;
    }
  }
  return func;
}

FailureOr<LLVM::LLVMFuncOp>
cloneLLVMSymbolsAndLookupFunc(ModuleOp curModule, ModuleOp parsedModule,
                              std::optional<llvm::StringRef> funcName) {
  LLVM::LLVMFuncOp parsedFunc = findExternalLLVMFunc(parsedModule, funcName);
  if (!parsedFunc)
    return failure();

  OpBuilder builder(curModule.getContext());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(curModule.getBody());
  IRMapping mapper;
  for (auto func : parsedModule.getOps<LLVM::LLVMFuncOp>()) {
    if (!curModule.lookupSymbol<LLVM::LLVMFuncOp>(func.getSymName()))
      builder.clone(*func, mapper);
  }
  for (auto global : parsedModule.getOps<LLVM::GlobalOp>()) {
    if (!curModule.lookupSymbol<LLVM::GlobalOp>(global.getSymName()))
      builder.clone(*global, mapper);
  }

  LLVM::LLVMFuncOp funcOp =
      curModule.lookupSymbol<LLVM::LLVMFuncOp>(parsedFunc.getSymName());
  if (!funcOp)
    return failure();
  return funcOp;
}

LogicalResult buildDSLRegionBodyFromLLVMFunc(TritonOpBuilder &builder,
                                             tle::DSLRegionOp dslRegionOp,
                                             LLVM::LLVMFuncOp funcOp) {
  OpBuilder &mlirBuilder = builder.getBuilder();
  OpBuilder::InsertionGuard guard(mlirBuilder);
  Location loc = dslRegionOp.getLoc();
  builder.setLastLoc(loc);

  Region &body = dslRegionOp.getBody();
  while (!body.empty())
    body.front().erase();

  SmallVector<Type> operandTys;
  operandTys.reserve(dslRegionOp.getNumOperands());
  for (Value operand : dslRegionOp.getOperands())
    operandTys.push_back(operand.getType());

  Block *entryBlock = mlirBuilder.createBlock(
      &body, {}, operandTys, SmallVector<Location>(operandTys.size(), loc));
  builder.setInsertionPointToStart(*entryBlock);

  TypeRange tgts = funcOp.getArgumentTypes();
  SmallVector<Value> callOperands;
  for (Value src : entryBlock->getArguments()) {
    SmallVector<Value> rets =
        tle::protocol::SignaturePattern::apply(builder, tgts, src);
    callOperands.append(std::move(rets));
  }

  builder.setInsertionPointToEnd(*entryBlock);
  LLVM::CallOp callOp = builder.create<LLVM::CallOp>(funcOp, callOperands);
  callOp.setAlwaysInline(true);

  Type retTy = funcOp.getFunctionType().getReturnType();
  SmallVector<Value> yields;
  if (isa<LLVM::LLVMVoidType>(retTy)) {
    if (dslRegionOp.getNumResults() == 0) {
      yields = {};
    } else {
      for (int32_t idx : dslRegionOp.getOutputOperandIndices())
        yields.push_back(entryBlock->getArgument(idx));
    }
  } else {
    SmallVector<Value> operands;
    if (dslRegionOp.getNumResults() == 0) {
      operands = {};
    } else if (dslRegionOp.getNumResults() == 1) {
      operands = callOp.getResults();
    } else {
      operands = flatten(
          builder, cast<TypedValue<LLVM::LLVMStructType>>(callOp.getResult()));
    }
    TypeRange outTys = dslRegionOp.getOutputs().getTypes();
    for (Value operand : operands) {
      SmallVector<Value> rets =
          tle::protocol::ReturnPattern::apply(builder, outTys, operand);
      yields.append(std::move(rets));
    }
  }

  builder.setLastLoc(loc);
  builder.create<tle::YieldOp>(yields);
  return success();
}

LogicalResult materializeDeferredDSLRegion(ModuleOp module, tle::DSLRegionOp op,
                                           llvm::StringRef llvmIr,
                                           llvm::StringRef externFuncName) {
  OwningOpRef<ModuleOp> parsedModule =
      parseLLVMModule(module.getContext(), llvmIr);
  if (!parsedModule)
    return failure();

  auto funcOpOrErr =
      cloneLLVMSymbolsAndLookupFunc(module, parsedModule.get(), externFuncName);
  if (failed(funcOpOrErr))
    return failure();

  TritonOpBuilder builder(module.getContext());
  builder.setLastLoc(op.getLoc());
  return buildDSLRegionBodyFromLLVMFunc(builder, op, *funcOpOrErr);
}

} // namespace mlir::triton::tle::raw
