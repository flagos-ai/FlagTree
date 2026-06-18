#include "ir.h"

#include "IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/AnalyzeReturnType.h"
#include "tle/utils/include/TleRawMaterialize.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace tle = triton::tle;

namespace {
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

tle::DSLRegionOp createDSLRegionOp(
    TritonOpBuilder &self, ArrayRef<Type> outputTys, ArrayRef<Value> operands,
    std::string_view regionDialect, std::string_view argDialect,
    ArrayRef<int64_t> aliasOperandIndices, std::string_view hint) {
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
  OwningOpRef<ModuleOp> module =
      tle::raw::parseLLVMModule(self.getContext(), text);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = tle::raw::findExternalLLVMFunc(module.get(), {});
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

tle::DSLRegionOp createTLERawRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint) {
  OwningOpRef<ModuleOp> module =
      tle::raw::parseLLVMModule(self.getContext(), text);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = tle::raw::findExternalLLVMFunc(module.get(), {});
  assert(func && "No function found in LLVM IR text");

  OpBuilder &builder = self.getBuilder();
  Operation *curOp = builder.getInsertionBlock()->getParentOp();
  while (curOp && curOp->getParentOp() && !isa<ModuleOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  ModuleOp curModule = cast<ModuleOp>(curOp);

  auto funcOpOrErr =
      tle::raw::cloneLLVMSymbolsAndLookupFunc(curModule, module.get(), {});
  assert(succeeded(funcOpOrErr));
  LLVM::LLVMFuncOp funcOp = *funcOpOrErr;

  Type retTy = funcOp.getFunctionType().getReturnType();
  SmallVector<Type> outputTys =
      isa<LLVM::LLVMVoidType>(retTy)
          ? SmallVector<Type>{}
          : llvm::map_to_vector(aliasOperandIndices, [&](int64_t idx) -> Type {
              return args[idx].getType();
            });

  SmallVector<Value> operands(args.begin(), args.end());
  tle::DSLRegionOp dslRegionOp =
      createDSLRegionOp(self, outputTys, operands, regionDialect, argDialect,
                        aliasOperandIndices, hint);
  assert(succeeded(
      tle::raw::buildDSLRegionBodyFromLLVMFunc(self, dslRegionOp, funcOp)));
  return dslRegionOp;
}

tle::DSLRegionOp createTLERawRegionDeferred(
    TritonOpBuilder &self, std::string_view sourceId,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint) {
  OpBuilder &builder = self.getBuilder();
  SmallVector<Type> outputTys =
      llvm::map_to_vector(aliasOperandIndices, [&](int64_t idx) -> Type {
        return args[idx].getType();
      });
  SmallVector<Value> operands(args.begin(), args.end());
  tle::DSLRegionOp dslRegionOp =
      createDSLRegionOp(self, outputTys, operands, regionDialect, argDialect,
                        aliasOperandIndices, hint);
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
