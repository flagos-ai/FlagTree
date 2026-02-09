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

static SmallVector<Type>
aggregationTypes(TritonOpBuilder &builder,
                 const SmallVector<Type> &unconvertTypes,
                 const SmallVector<Type> &convertTypes) {
  SmallVector<Type> resultTypes;
  TypeRange tgts = convertTypes;
  for (Type singletype : unconvertTypes) {
    if (auto ptrType = dyn_cast<RankedTensorType>(singletype)) {
      size_t rank = ptrType.getRank();
      Type allocatedPtrTy = tgts[0];
      Type alignedPtrTy = tgts[1];
      Type offsetTy = tgts[2];
      Type sizeElemTy = tgts[3];
      Type strideElemTy = tgts[3 + rank];
      auto sizesArrayTy = LLVM::LLVMArrayType::get(sizeElemTy, rank);
      auto stridesArrayTy = LLVM::LLVMArrayType::get(strideElemTy, rank);
      SmallVector<Type> fieldTys = {
          allocatedPtrTy, alignedPtrTy, offsetTy, sizesArrayTy, stridesArrayTy,
      };
      resultTypes.push_back(LLVM::LLVMStructType::getLiteral(
          builder.getContext(), fieldTys, /*packed=*/false));
    } else {
      resultTypes.push_back(std::move(tgts.front()));
      tgts = tgts.drop_front();
    }
  }
  return resultTypes;
}
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
  SmallVector<Type> funcArgTypes =
      llvm::map_to_vector(func.getArguments(), [](BlockArgument arg) -> Type {
        return arg.getType();
      });

  SmallVector<Value> converted_inputs;
  {
    OpBuilder::InsertionGuard guard(builder);
    TypeRange tgts = funcArgTypes;
    SmallVector<Value> rets;
    for (Value src : inputs) {
      SmallVector<Value> rets =
          tle::protocol::SignaturePattern::apply(self, tgts, src);
      converted_inputs.append(std::move(rets));
    }
  }
  SmallVector<Value> converted_outputs;
  {
    OpBuilder::InsertionGuard guard(builder);
    TypeRange tgts = funcArgTypes;
    SmallVector<Value> rets;
    for (Value src : outputs) {
      SmallVector<Value> rets =
          tle::protocol::SignaturePattern::apply(self, tgts, src);
      converted_outputs.append(std::move(rets));
    }
  }
  SmallVector<Type> outputTys = llvm::map_to_vector(
      outputs, [](Value value) -> Type { return value.getType(); });
  SmallVector<Value> operands =
      llvm::to_vector(llvm::concat<Value>(converted_outputs, converted_inputs));

  SmallVector<Type> dslOutputTys = llvm::map_to_vector(
      converted_outputs, [](Value value) -> Type { return value.getType(); });
  auto outStructTy = aggregationTypes(self, outputTys, dslOutputTys);
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

      SmallVector<Value> ops =
          llvm::map_to_vector(newBlock->getArguments(),
                              [](BlockArgument arg) -> Value { return arg; });
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
        SmallVector<Value> operands, yields;
        if (dslRegionOp.getNumResults() == 0) {
          operands = {};
        } else if (dslRegionOp.getNumResults() == 1) {
          operands = {mapper.lookup(returnOp.getArg())};
        } else {
          operands = flatten(self, cast<TypedValue<LLVM::LLVMStructType>>(
                                       mapper.lookup(returnOp.getArg())));
        }
        TypeRange tgts = dslRegionOp.getOutputs().getTypes();
        builder.create<tle::YieldOp>(operation.getLoc(), operands);
      } else {
        builder.clone(operation, mapper);
      }
    }
  }

  builder.setInsertionPointAfter(dslRegionOp);
  SmallVector<Value> finalResults;
  TypeRange tgts = outputTys;
  for (Value result : dslRegionOp.getResults()) {
    SmallVector<Value> rets =
        tle::protocol::ReturnPattern::apply(self, tgts, result);
    finalResults.append(std::move(rets));
  }
  return finalResults;
}
