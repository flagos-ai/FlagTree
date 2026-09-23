#ifdef __ILUVATAR_TLE__

#include "AnalyzeReturnType.h"
#include "IR/Dialect.h"
#include "TleRawMaterialize.h"
#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <vector>

using namespace mlir;
namespace iluvatar_tle = triton::iluvatar_tle;

namespace {
StringAttr getOptionalStringAttr(OpBuilder &builder, std::string_view value) {
  if (value.empty())
    return StringAttr();
  return builder.getStringAttr(value);
}

std::optional<llvm::StringRef> getOptionalFuncName(std::string_view value) {
  if (value.empty())
    return std::nullopt;
  return llvm::StringRef(value.data(), value.size());
}

void setDeferredMetadataAttrs(iluvatar_tle::DSLRegionOp op, OpBuilder &builder,
                              std::string_view sourceId,
                              std::string_view dsl_file_name,
                              std::string_view extern_func_name) {
  if (!sourceId.empty())
    op->setAttr("tle_raw.source_id", builder.getStringAttr(sourceId));
  if (!dsl_file_name.empty())
    op->setAttr("tle_raw.dsl_file_name", builder.getStringAttr(dsl_file_name));
  if (!extern_func_name.empty())
    op->setAttr("tle_raw.extern_func_name",
                builder.getStringAttr(extern_func_name));
}

iluvatar_tle::DSLRegionOp createDSLRegionOp(
    TritonOpBuilder &self, ArrayRef<Type> outputTys, ArrayRef<Value> operands,
    std::string_view regionDialect, std::string_view argDialect,
    ArrayRef<int64_t> aliasOperandIndices, std::string_view hint) {
  OpBuilder &builder = self.getBuilder();
  SmallVector<int32_t> outputIndices(aliasOperandIndices.begin(),
                                     aliasOperandIndices.end());
  return self.create<iluvatar_tle::DSLRegionOp>(
      outputTys, operands, regionDialect, argDialect, outputIndices,
      getOptionalStringAttr(builder, hint));
}
} // namespace

std::vector<int64_t> computeAliasOperandIndices(TritonOpBuilder &self,
                                                std::string_view text,
                                                const std::vector<Value> &args,
                                                std::string_view funcName) {
  OwningOpRef<ModuleOp> module =
      iluvatar_tle::raw::parseLLVMModule(self.getContext(), text);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = iluvatar_tle::raw::findExternalLLVMFunc(
      module.get(), getOptionalFuncName(funcName));
  assert(func && "No function found in LLVM IR text");

  SmallVector<int64_t> funcArgToDslArg =
      iluvatar_tle::data_analyze::computeFuncArgToDslArg(args);

  auto funcType = func.getFunctionType();
  Type retTy = funcType.getReturnType();
  if (isa<LLVM::LLVMVoidType>(retTy))
    return {};

  auto aliasesOrFailure = iluvatar_tle::data_analyze::analyzeFuncReturnAliases(
      func, funcArgToDslArg);
  assert(succeeded(aliasesOrFailure));
  SmallVector<int64_t> result = *aliasesOrFailure;
  return std::vector<int64_t>(result.begin(), result.end());
}

iluvatar_tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
                             std::string_view regionDialect,
                             std::string_view argDialect,
                             const std::vector<Value> &args,
                             const std::vector<int64_t> &aliasOperandIndices,
                             std::string_view hint, std::string_view funcName) {
  OwningOpRef<ModuleOp> module =
      iluvatar_tle::raw::parseLLVMModule(self.getContext(), text);
  assert(module && "Failed to parse LLVM IR text");
  LLVM::LLVMFuncOp func = iluvatar_tle::raw::findExternalLLVMFunc(
      module.get(), getOptionalFuncName(funcName));
  assert(func && "No function found in LLVM IR text");

  OpBuilder &builder = self.getBuilder();
  Operation *curOp = builder.getInsertionBlock()->getParentOp();
  while (curOp && curOp->getParentOp() && !isa<ModuleOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  ModuleOp curModule = cast<ModuleOp>(curOp);

  auto funcOpOrErr = iluvatar_tle::raw::cloneLLVMSymbolsAndLookupFunc(
      curModule, module.get(), getOptionalFuncName(funcName));
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
  iluvatar_tle::DSLRegionOp dslRegionOp =
      createDSLRegionOp(self, outputTys, operands, regionDialect, argDialect,
                        aliasOperandIndices, hint);
  assert(succeeded(iluvatar_tle::raw::buildDSLRegionBodyFromLLVMFunc(
      self, dslRegionOp, funcOp)));
  return dslRegionOp;
}

iluvatar_tle::DSLRegionOp createTLERawRegionDeferred(
    TritonOpBuilder &self, std::string_view sourceId,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint,
    std::string_view dsl_file_name, std::string_view extern_func_name) {
  OpBuilder &builder = self.getBuilder();
  SmallVector<Type> outputTys =
      llvm::map_to_vector(aliasOperandIndices, [&](int64_t idx) -> Type {
        return args[idx].getType();
      });
  SmallVector<Value> operands(args.begin(), args.end());
  iluvatar_tle::DSLRegionOp dslRegionOp =
      createDSLRegionOp(self, outputTys, operands, regionDialect, argDialect,
                        aliasOperandIndices, hint);
  setDeferredMetadataAttrs(dslRegionOp, builder, sourceId, dsl_file_name,
                           extern_func_name);

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
  builder.create<iluvatar_tle::YieldOp>(self.getLastLoc(), yields);
  return dslRegionOp;
}

#endif // __ILUVATAR_TLE__
