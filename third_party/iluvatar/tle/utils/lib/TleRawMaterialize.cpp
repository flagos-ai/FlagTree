#include "TleRawMaterialize.h"
#include "Protocol.h"
#include "ir.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

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

namespace mlir::triton::iluvatar_tle::raw {

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

FailureOr<llvm::StringMap<Operation *>> renameLLVMSymbols(
    ModuleOp module,
    llvm::function_ref<StringAttr(Operation *)> targetNameGenerator) {

  llvm::StringMap<Operation *> originals;
  llvm::StringSet<> occupied, finalNames;
  SmallVector<std::pair<Operation *, StringAttr>> changes;

  for (Operation &op : module.getBody()->getOperations()) {
    auto name = op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (name)
      occupied.insert(name.getValue());
    if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp>(op))
      continue;

    auto finalName = targetNameGenerator(&op);
    if (!finalName || finalName.getValue().empty() ||
        !finalNames.insert(finalName.getValue()).second) {
      return failure();
    }
    if (Operation *other = module.lookupSymbol(finalName.getValue())) {
      if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp>(other)) {
        return failure();
      }
    }
    occupied.insert(finalName);
    originals[name.getValue()] = &op;
    if (name != finalName)
      changes.emplace_back(&op, finalName);
  }

  unsigned temporaryIndex = 0;
  for (auto [op, finalName] : changes) {
    std::string temporary;
    do {
      temporary = "__tle_raw_rename_tmp_" + std::to_string(temporaryIndex++);
    } while (!occupied.insert(temporary).second);
    auto temporaryName = StringAttr::get(module.getContext(), temporary);
    if (failed(SymbolTable::replaceAllSymbolUses(op, temporaryName, module))) {
      op->emitError("cannot update raw symbol references");
      return failure();
    }
    SymbolTable::setSymbolName(op, temporaryName);
  }

  for (auto [op, finalName] : changes) {
    if (failed(SymbolTable::replaceAllSymbolUses(op, finalName, module))) {
      op->emitError("cannot update raw symbol references");
      return failure();
    }
    SymbolTable::setSymbolName(op, finalName);
  }
  return originals;
}

FailureOr<IRMapping> cloneLLVMSymbols(ModuleOp dstModule, ModuleOp srcModule) {
  // Check existing users' type and address space before changing the module.
  for (auto global : srcModule.getOps<LLVM::GlobalOp>()) {
    auto target = dstModule.lookupSymbol<LLVM::GlobalOp>(global.getSymName());
    if (target && (target.getGlobalType() != global.getGlobalType() ||
                   target.getAddrSpace() != global.getAddrSpace()))
      return global.emitOpError(
                 "incompatible type or address space for global @")
             << global.getSymName();
  }
  OpBuilder builder(dstModule.getContext());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(dstModule.getBody());
  IRMapping mapper;
  for (auto func : srcModule.getOps<LLVM::LLVMFuncOp>()) {
    if (auto target =
            dstModule.lookupSymbol<LLVM::LLVMFuncOp>(func.getSymName()))
      mapper.map(func.getOperation(), target.getOperation());
    else
      builder.clone(*func, mapper);
  }
  for (auto global : srcModule.getOps<LLVM::GlobalOp>()) {
    auto target = dstModule.lookupSymbol<LLVM::GlobalOp>(global.getSymName());
    if (!target) {
      builder.clone(*global, mapper);
      continue;
    }
    mapper.map(global.getOperation(), target.getOperation());
    auto hasInitializer = [](LLVM::GlobalOp op) {
      return op.getValueOrNull() || !op.getInitializerRegion().empty();
    };
    if (!hasInitializer(target) && hasInitializer(global)) {
      // Earlier imports may still hold the declaration's Operation *.
      target->setAttrs(global->getAttrDictionary());
      target->setLoc(global.getLoc());
      Region &initializer = global.getInitializerRegion();
      if (!initializer.empty())
        initializer.cloneInto(&target.getInitializerRegion(), mapper);
    }
  }
  return mapper;
}

FailureOr<LLVM::LLVMFuncOp>
cloneLLVMSymbolsAndLookupFunc(ModuleOp curModule, ModuleOp parsedModule,
                              std::optional<llvm::StringRef> funcName) {
  LLVM::LLVMFuncOp parsedFunc = findExternalLLVMFunc(parsedModule, funcName);
  if (!parsedFunc)
    return failure();
  if (failed(cloneLLVMSymbols(curModule, parsedModule)))
    return failure();
  LLVM::LLVMFuncOp funcOp =
      curModule.lookupSymbol<LLVM::LLVMFuncOp>(parsedFunc.getSymName());
  if (!funcOp)
    return failure();
  return funcOp;
}

LogicalResult buildDSLRegionBodyFromLLVMFunc(TritonOpBuilder &builder,
                                             DSLRegionOp dslRegionOp,
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
        protocol::SignaturePattern::apply(builder, tgts, src);
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
          protocol::ReturnPattern::apply(builder, outTys, operand);
      yields.append(std::move(rets));
    }
  }

  builder.setLastLoc(loc);
  builder.create<YieldOp>(yields);
  return success();
}

LogicalResult materializeDeferredDSLRegion(ModuleOp module, DSLRegionOp op,
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

} // namespace mlir::triton::iluvatar_tle::raw
