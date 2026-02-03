#include "IR/Dialect.h"
#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "tle/dialect/include/IR/Dialect.h"
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
SmallVector<Value> createTLERawRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs) {
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
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
      if (&op != func.getOperation() &&
          (!isa<SymbolOpInterface>(op) ||
           (isa<SymbolOpInterface>(op) &&
            !curModule.lookupSymbol(cast<SymbolOpInterface>(op).getName())))) {
        builder.clone(op);
      }
    }
  }
    // Convert outputs to LLVM types
  SmallVector<Value> converted_outputs;
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Type> funcArgTypes = llvm::map_to_vector(
        func.getArguments(), [](BlockArgument arg) -> Type { return arg.getType(); });
    TypeRange tgts = funcArgTypes;
    for (Value src : outputs) {
      SmallVector<Value> rets =
          tle::protocol::SignaturePattern::apply(self, tgts, src);
      converted_outputs.append(std::move(rets));
    }
  }
  SmallVector<Type> outputTys = llvm::map_to_vector(
       outputs, [](Value value) -> Type { return value.getType(); });
  // SmallVector<Value> operands = llvm::to_vector
  // Convert inputs to LLVM types
  SmallVector<Value> converted_inputs;
  {
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Type> funcArgTypes = llvm::map_to_vector(
        func.getArguments(), [](BlockArgument arg) -> Type { return arg.getType(); });
    TypeRange tgts = funcArgTypes;
    for (Value src : inputs) {
      SmallVector<Value> rets =
          tle::protocol::SignaturePattern::apply(self, tgts, src);
      converted_inputs.append(std::move(rets));
    }
  }

  // Combine converted outputs and inputs
  SmallVector<Value> operands = llvm::to_vector(
      llvm::concat<Value>(converted_outputs, converted_inputs));

  // DSLRegionOp 返回完整的 LLVM struct（不拆解）
  SmallVector<Type> dslOutputTys = llvm::map_to_vector(
        converted_outputs, [](Value value) -> Type { return value.getType(); });
  for (Type dsloutput : dslOutputTys) {
    llvm::outs() << "=== outputTy: " << dsloutput << "\n";
  }
  auto outStructTy = LLVM::LLVMStructType::getLiteral(self.getContext(), dslOutputTys, /*packed=*/false);
  // if (func.getNumResults() > 0) {
  //   dslOutputTys.push_back(func.getResultTypes()[0]);
  // }

  tle::DSLRegionOp dslRegionOp =
      self.create<tle::DSLRegionOp>(outStructTy, operands);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;
  for (auto [idx, oldBlock] : enumerate(func.getBlocks())) {
    if (idx == 0) {
      Block *newBlock = builder.createBlock(
          &body, {}, operandTys,
          SmallVector<Location>(operandTys.size(), self.getLastLoc()));
      builder.setInsertionPointToStart(newBlock);
      // ValueRange args = func.getArguments();
      // TypeRange tgts = args.getTypes();
      SmallVector<Value> ops = llvm::map_to_vector(
          newBlock->getArguments(), [](BlockArgument arg) -> Value { return arg; });
      // SmallVector<Value> ops = {};
      // for (Value src : newBlock->getArguments()) {
      //   SmallVector<Value> rets =
      //       tle::protocol::SignaturePattern::apply(self, tgts, src);
      //   ops.append(std::move(rets));
      // }
      for (auto [arg, op] : zip_equal(func.getArguments(), ops)) {
        mapper.map(arg, op);
      }
      mapper.map(&oldBlock, newBlock);
    } else {
      Block *newBlock = builder.createBlock(
          &body, {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), self.getLastLoc()));
      for (auto [oldArg, newArg] :
           zip_equal(oldBlock.getArguments(), newBlock->getArguments())) {
        mapper.map(oldArg, newArg);
      }
      mapper.map(&oldBlock, newBlock);
    }
  }
  for (auto [oldBlock, newBlock] :
       zip_equal(func.getBlocks(), body.getBlocks())) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (LLVM::ReturnOp returnOp = dyn_cast<LLVM::ReturnOp>(operation)) {
        // 直接 yield 整个 LLVM struct（或其他返回值）
        SmallVector<Value> yields;
        if (dslRegionOp.getNumResults() > 0) {
          yields.push_back(cast<TypedValue<LLVM::LLVMStructType>>(mapper.lookup(returnOp.getArg())));
        }
        llvm::outs()<<"Creating tle.yield with "<<yields.size()<<" values\n";
        builder.create<tle::YieldOp>(operation.getLoc(), yields);
      } else {
        builder.clone(operation, mapper);
      }
    }
  }

  // 在 dsl_region 外部对返回的 struct 调用 ReturnPattern 创建 tle.pack
  SmallVector<Value> finalResults;
  if (dslRegionOp.getNumResults() > 0) {
    TypeRange tgts = outputTys;  // 原始的 tensor 输出类型
    llvm::outs() << "=== dslRegionOp.getNumResults() = " << dslRegionOp.getNumResults() << "\n";
    llvm::outs() << "=== outputTys.size() = " << outputTys.size() << "\n";
    for (Value result : dslRegionOp.getResults()) {
      llvm::outs() << "=== Processing result type: " << result.getType() << "\n";
      SmallVector<Value> rets =
          tle::protocol::ReturnPattern::apply(self, tgts, result);
      llvm::outs() << "=== rets.size() = " << rets.size() << "\n";
      finalResults.append(std::move(rets));
    }
  }
  llvm::outs() << "=== finalResults.size() = " << finalResults.size() << "\n";

  return finalResults;
}