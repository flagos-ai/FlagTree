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

// Helper: get the DSL arg index for a BlockArgument via funcArgToDslArg map.
// Returns -1 if out of range.
static int64_t getDslArgIdx(BlockArgument blockArg,
                            ArrayRef<int64_t> funcArgToDslArg) {
  int64_t funcIdx = blockArg.getArgNumber();
  if (funcIdx >= 0 && funcIdx < (int64_t)funcArgToDslArg.size())
    return funcArgToDslArg[funcIdx];
  return -1;
}

// Trace a single Value back to its source DSL arg index.
// A memref descriptor is rebuilt from multiple LLVM func args that all map
// to the same DSL arg. We check consistency at the DSL arg level.
// Returns the DSL arg index, or -1 if unknown.
static int64_t traceToDslArg(Value val, ArrayRef<int64_t> funcArgToDslArg) {
  if (auto blockArg = dyn_cast<BlockArgument>(val))
    return getDslArgIdx(blockArg, funcArgToDslArg);

  auto insertOp = val.getDefiningOp<LLVM::InsertValueOp>();
  if (!insertOp)
    return -1;

  int64_t aliasedDslIdx = -1;
  Operation *current = insertOp;
  while (auto ivOp = dyn_cast_or_null<LLVM::InsertValueOp>(current)) {
    Value insertedVal = ivOp.getValue();
    int64_t thisDslIdx = -1;
    if (auto evOp = insertedVal.getDefiningOp<LLVM::ExtractValueOp>()) {
      if (auto blockArg = dyn_cast<BlockArgument>(evOp.getContainer()))
        thisDslIdx = getDslArgIdx(blockArg, funcArgToDslArg);
      else
        return -1;
    } else if (auto blockArg = dyn_cast<BlockArgument>(insertedVal)) {
      thisDslIdx = getDslArgIdx(blockArg, funcArgToDslArg);
    } else {
      return -1;
    }

    if (thisDslIdx < 0)
      return -1;
    if (aliasedDslIdx == -1)
      aliasedDslIdx = thisDslIdx;
    else if (aliasedDslIdx != thisDslIdx)
      return -1;

    current = ivOp.getContainer().getDefiningOp();
  }

  if (!current ||
      !(isa<LLVM::UndefOp>(current) || isa<LLVM::PoisonOp>(current)))
    return -1;
  return aliasedDslIdx;
}

// Analyze the LLVM function's return to determine per-result alias info.
// Returns a vector of DSL arg indices (one per DSLRegionOp result).
// -1 means unknown / no alias.
//
// numResults == 0: void → empty
// numResults == 1: trace the whole return value
// numResults >  1: outer struct, trace each top-level field independently
static SmallVector<int64_t>
analyzeFuncReturnAliases(LLVM::LLVMFuncOp func, size_t numResults,
                         ArrayRef<int64_t> funcArgToDslArg) {
  if (numResults == 0)
    return {};

  LLVM::ReturnOp retOp = nullptr;
  func.walk([&](LLVM::ReturnOp op) { retOp = op; });
  if (!retOp || retOp.getNumOperands() == 0)
    return SmallVector<int64_t>(numResults, -1);

  Value retVal = retOp.getOperand(0);

  // Single result: trace the whole return value
  if (numResults == 1)
    return {traceToDslArg(retVal, funcArgToDslArg)};

  // Multiple results: outer struct, each top-level field traced independently.
  SmallVector<int64_t> result(numResults, -1);

  // If the whole return is a BlockArgument, all fields alias it
  if (auto blockArg = dyn_cast<BlockArgument>(retVal)) {
    int64_t dslIdx = getDslArgIdx(blockArg, funcArgToDslArg);
    return SmallVector<int64_t>(numResults, dslIdx);
  }

  Value current = retVal;
  while (auto ivOp = current.getDefiningOp<LLVM::InsertValueOp>()) {
    ArrayRef<int64_t> position = ivOp.getPosition();
    if (position.size() == 1) {
      int64_t fieldIdx = position[0];
      if (fieldIdx >= 0 && fieldIdx < (int64_t)numResults)
        result[fieldIdx] = traceToDslArg(ivOp.getValue(), funcArgToDslArg);
    }
    current = ivOp.getContainer();
  }
  return result;
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
tle::DSLRegionOp createTLERawRegionByLLVMFunc(TritonOpBuilder &self,
                                              std::string_view text,
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
  OpBuilder &builder = self.getBuilder();
  Operation *curOp = builder.getInsertionBlock()->getParentOp();
  while (curOp && curOp->getParentOp() && !isa<ModuleOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  ModuleOp curModule = cast<ModuleOp>(curOp);
  {
    llvm::outs() << " curModule \n";
    curModule->print(llvm::outs());
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

  // Infer output types from LLVM function's return type.
  // For struct returns (lowered memref descriptors), map back to the
  // corresponding Triton IR type (RankedTensorType) from the operands,
  // since ReturnPattern expects Triton-level target types.
  // For scalar returns (f32, i32, etc.), use the type directly.
  SmallVector<Type> outputTys;
  Type retTy = funcOp.getFunctionType().getReturnType();
  if (!isa<LLVM::LLVMVoidType>(retTy)) {
    if (isa<LLVM::LLVMStructType>(retTy)) {
      for (const Value &arg : args) {
        if (auto tensorTy = dyn_cast<RankedTensorType>(arg.getType())) {
          outputTys.push_back(tensorTy);
          break;
        }
      }
    } else {
      outputTys.push_back(retTy);
    }
  }

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
  SmallVector<int64_t> funcArgToDslArg;
  int64_t dslArgIdx = 0;
  for (Value src : newBlock->getArguments()) {
    size_t before = ops.size();
    SmallVector<Value> rets =
        tle::protocol::SignaturePattern::apply(self, tgts, src);
    ops.append(std::move(rets));
    for (size_t i = before; i < ops.size(); ++i)
      funcArgToDslArg.push_back(dslArgIdx);
    dslArgIdx++;
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
  // Analyze alias: which DSLRegionOp result aliases which DSL operand
  size_t numResults = dslRegionOp.getNumResults();
  SmallVector<int64_t> aliasOperandIndices =
      analyzeFuncReturnAliases(func, numResults, funcArgToDslArg);
  dslRegionOp->setAttr(
      "tle.alias_operand_indices",
      DenseI64ArrayAttr::get(builder.getContext(), aliasOperandIndices));

  return dslRegionOp;
}
