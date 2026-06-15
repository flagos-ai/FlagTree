#include "IR/Dialect.h"
#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/AnalyzeReturnType.h"
#include "tle/utils/include/Protocol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;
namespace tle = triton::tle;

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

StringAttr getOptionalStringAttr(OpBuilder &builder, std::string_view value) {
  if (value.empty())
    return StringAttr();
  return builder.getStringAttr(value);
}

void setDeferredMetadataAttrs(tle::DSLRegionOp op, OpBuilder &builder,
                              std::string_view sourceId) {
  if (!sourceId.empty())
    op->setAttr("tle_raw.source_id", builder.getStringAttr(sourceId));
}

tle::DSLRegionOp createDSLRegionOp(TritonOpBuilder &self,
                                   ArrayRef<Type> outputTys,
                                   ArrayRef<Value> operands,
                                   std::string_view regionDialect,
                                   std::string_view argDialect,
                                   ArrayRef<int64_t> aliasOperandIndices,
                                   std::string_view hint) {
  OpBuilder &builder = self.getBuilder();
  SmallVector<int32_t> outputIndices(aliasOperandIndices.begin(),
                                     aliasOperandIndices.end());
  return self.create<tle::DSLRegionOp>(outputTys, operands, regionDialect,
                                       argDialect, outputIndices,
                                       getOptionalStringAttr(builder, hint));
}
} // namespace

std::vector<int64_t>
computeAliasOperandIndices(TritonOpBuilder &self, std::string_view text,
                           const std::vector<Value> &args) {
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = nullptr;
  for (auto op : module->getOps<LLVM::LLVMFuncOp>()) {
    if (!op.empty() && op.getLinkage() != LLVM::Linkage::Internal) {
      if (func) {
        llvm_unreachable("Multiple functions found in LLVM IR text");
      } else {
        func = op;
      }
    }
  }
  assert(func && "No function found in LLVM IR text");

  SmallVector<int64_t> funcArgToDslArg =
      tle::data_analyze::computeFuncArgToDslArg(args);

  auto funcType = func.getFunctionType();
  Type retTy = funcType.getReturnType();
  if (isa<LLVM::LLVMVoidType>(retTy))
    return {};

  auto aliasesOrFailure =
      tle::data_analyze::analyzeFuncReturnAliases(func, funcArgToDslArg);
  assert(succeeded(aliasesOrFailure));
  SmallVector<int64_t> result = *aliasesOrFailure;
  return std::vector<int64_t>(result.begin(), result.end());
}

tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
                             std::string_view regionDialect,
                             std::string_view argDialect,
                             const std::vector<Value> &args,
                             const std::vector<int64_t> &aliasOperandIndices,
                             std::string_view hint) {
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = nullptr;
  for (auto op : module->getOps<LLVM::LLVMFuncOp>()) {
    if (!op.empty() && op.getLinkage() != LLVM::Linkage::Internal) {
      if (func) {
        llvm_unreachable("Multiple functions found in LLVM IR text");
      } else {
        func = op;
      }
    }
  }
  assert(func && "No function found in LLVM IR text");
  OpBuilder &builder = self.getBuilder();
  Operation *curOp = builder.getInsertionBlock()->getParentOp();
  while (curOp && curOp->getParentOp() && !isa<ModuleOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  ModuleOp curModule = cast<ModuleOp>(curOp);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(curModule.getBody());
    for (Operation &op : module->getOps()) {
      if ((!isa<SymbolOpInterface>(op) ||
           (isa<SymbolOpInterface>(op) &&
            !curModule.lookupSymbol(cast<SymbolOpInterface>(op).getName()))) &&
          !isa<LLVM::ModuleFlagsOp>(op)) {
        builder.clone(op);
      }
    }
  }
  LLVM::LLVMFuncOp funcOp =
      curModule.lookupSymbol<LLVM::LLVMFuncOp>(func.getSymName());
  assert(funcOp && "callee function not found in current module");

  Type retTy = funcOp.getFunctionType().getReturnType();
  SmallVector<Type> outputTys =
      isa<LLVM::LLVMVoidType>(retTy)
          ? SmallVector<Type>{}
          : llvm::map_to_vector(aliasOperandIndices, [&](int64_t idx) -> Type {
              return args[idx].getType();
            });

  SmallVector<Value> operands(args.begin(), args.end());
  tle::DSLRegionOp dslRegionOp = createDSLRegionOp(
      self, outputTys, operands, regionDialect, argDialect, aliasOperandIndices,
      hint);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;
  Block *newBlock = builder.createBlock(
      &body, {}, operandTys,
      SmallVector<Location>(operandTys.size(), self.getLastLoc()));
  builder.setInsertionPointToStart(newBlock);
  ValueRange funcArgs = func.getArguments();
  TypeRange tgts = funcArgs.getType();
  SmallVector<Value> ops = {};
  for (Value src : newBlock->getArguments()) {
    SmallVector<Value> rets =
        tle::protocol::SignaturePattern::apply(self, tgts, src);
    ops.append(std::move(rets));
  }
  for (auto [funcArg, op] : zip_equal(func.getArguments(), ops)) {
    mapper.map(funcArg, op);
  }
  builder.setInsertionPointToEnd(newBlock);
  LLVM::CallOp callOp = self.create<LLVM::CallOp>(funcOp, ops);
  callOp.setAlwaysInline(true);

  tgts = dslRegionOp.getOutputs().getTypes();
  for (auto &oldBlock : func.getBlocks()) {
    for (Operation &operation : oldBlock.getOperations()) {
      if (LLVM::ReturnOp returnOp = dyn_cast<LLVM::ReturnOp>(operation)) {
        SmallVector<Value> operands, yields;
        if (dslRegionOp.getNumResults() == 0) {
          operands = {};
        } else if (dslRegionOp.getNumResults() == 1) {
          operands = callOp.getResults();
        } else {
          operands = flatten(
              self, cast<TypedValue<LLVM::LLVMStructType>>(callOp.getResult()));
        }
        TypeRange tgts = dslRegionOp.getOutputs().getTypes();
        for (Value operand : operands) {
          SmallVector<Value> rets =
              tle::protocol::ReturnPattern::apply(self, tgts, operand);
          yields.append(std::move(rets));
        }
        builder.create<tle::YieldOp>(operation.getLoc(), yields);
      }
    }
  }
  return dslRegionOp;
}

tle::DSLRegionOp
createTLERawRegionDeferred(TritonOpBuilder &self, std::string_view sourceId,
                           std::string_view regionDialect,
                           std::string_view argDialect,
                           const std::vector<Value> &args,
                           const std::vector<int64_t> &aliasOperandIndices,
                           std::string_view hint) {
  OpBuilder &builder = self.getBuilder();
  SmallVector<Type> outputTys = llvm::map_to_vector(
      aliasOperandIndices,
      [&](int64_t idx) -> Type { return args[idx].getType(); });
  SmallVector<Value> operands(args.begin(), args.end());
  tle::DSLRegionOp dslRegionOp = createDSLRegionOp(
      self, outputTys, operands, regionDialect, argDialect, aliasOperandIndices,
      hint);
  setDeferredMetadataAttrs(dslRegionOp, builder, sourceId);

  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  Block *newBlock = builder.createBlock(
      &body, {}, operandTys,
      SmallVector<Location>(operandTys.size(), self.getLastLoc()));
  builder.setInsertionPointToStart(newBlock);
  SmallVector<Value> yields;
  for (int64_t idx : aliasOperandIndices)
    yields.push_back(newBlock->getArgument(idx));
  builder.create<tle::YieldOp>(self.getLastLoc(), yields);
  return dslRegionOp;
}
