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
} // namespace

// Create a DSLRegionOp that wraps an LLVM function, performing type conversion
// from Triton IR types to LLVM types based on EDSL function declarations.
//
// Overview:
// 1. Parse the LLVM IR text and extract the target function using Triton's MLIR
// context
// 2. Create a DSLRegionOp with EDSL function parameter types stored in
// attributes
// 3. Perform argument type conversion: TT IR types -> LLVM types (via extract
// operations)
//    - DSLRegionOp's operands are TT IR types (tensor, pointer, scalar)
//    - EDSL function declarations (stored in edsl_param_types attribute)
//    specify expected types
//    - LLVM function arguments are already in LLVM types
//    - We need to verify consistency: TT type -> EDSL param type -> LLVM func
//    arg type
//
// Example type conversion for tensor:
//   - TT IR: tensor<128xi32> (RankedTensorType)
//   - EDSL param type: "memref<?xi32, 3>" (stored in edsl_param_types
//   attribute)
//   - LLVM func: 5 args = allocated_ptr<3>, aligned_ptr<3>, offset, size[0],
//   stride[0]
//   - Conversion: Extract tensor into 5 LLVM values using
//   ExtractAllocatedPtrOp, etc.
//
// Example type conversion for scalar:
//   - TT IR: i32 (IntegerType)
//   - EDSL param type: "i32"
//   - LLVM func: 1 arg = i32
//   - Conversion: Use block argument directly
// Analyze the LLVM IR text and compute alias operand indices without creating
// any ops. This is exposed as a separate pybinding so Python can obtain
// aliased_args before calling createTLERawRegionByLLVMFunc.
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
                             const std::vector<Value> &args,
                             const std::vector<int64_t> &aliasOperandIndices) {
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

  // Use the externally provided aliasOperandIndices to determine output types.
  Type retTy = funcOp.getFunctionType().getReturnType();
  SmallVector<Type> outputTys =
      isa<LLVM::LLVMVoidType>(retTy)
          ? SmallVector<Type>{}
          : llvm::map_to_vector(aliasOperandIndices, [&](int64_t idx) -> Type {
              return args[idx].getType();
            });

  SmallVector<Value> operands(args.begin(), args.end());
  tle::DSLRegionOp dslRegionOp =
      self.create<tle::DSLRegionOp>(outputTys, operands);
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
