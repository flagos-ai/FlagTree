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

//===----------------------------------------------------------------------===//
// Lightweight worklist-based DSL arg origin propagation.
//
// For each Value in the function, computes which DSL arg it originates from.
// Three states per Value:
//   kUninitialized (-2): not yet visited
//   >= 0:               originates from DSL arg N
//   kConflict (-1):     multiple origins or non-DSL-arg origin
//
// The propagation is a simple forward pass: initialize block args from
// funcArgToDslArg, then iterate over all ops in program order. For each op
// result, join the origins of all operands. Repeat until fixpoint.
//===----------------------------------------------------------------------===//
static constexpr int64_t kUninitialized = -2;
static constexpr int64_t kConflict = -1;

static int64_t joinOrigin(int64_t lhs, int64_t rhs) {
  if (lhs == kUninitialized)
    return rhs;
  if (rhs == kUninitialized)
    return lhs;
  if (lhs == kConflict || rhs == kConflict)
    return kConflict;
  if (lhs == rhs)
    return lhs;
  return kConflict;
}

// Run forward origin propagation on a function.
// Returns a map from Value to DSL arg origin index.
static DenseMap<Value, int64_t>
computeDslArgOrigins(LLVM::LLVMFuncOp func,
                     ArrayRef<int64_t> funcArgToDslArg) {
  DenseMap<Value, int64_t> origins;

  // Initialize all block arguments
  for (Block &block : func.getBlocks()) {
    for (BlockArgument arg : block.getArguments()) {
      origins[arg] = getDslArgIdx(arg, funcArgToDslArg);
    }
  }

  // Iterate until fixpoint
  int iteration = 0;
  bool changed = true;
  while (changed) {
    iteration++;
    changed = false;
    for (Block &block : func.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        // undef/poison: leave as uninitialized (neutral in join)
        if (isa<LLVM::UndefOp, LLVM::PoisonOp>(op)) {
          for (Value result : op.getResults()) {
            if (!origins.count(result)) {
              origins[result] = kUninitialized;
            }
          }
          continue;
        }

        // Compute joined origin of all operands
        int64_t opOrigin = kUninitialized;
        if (op.getNumOperands() == 0) {
          // No operands (constants, etc.) → conflict
          opOrigin = kConflict;
        } else {
          for (Value operand : op.getOperands()) {
            int64_t operandOrigin = kUninitialized;
            auto it = origins.find(operand);
            if (it != origins.end())
              operandOrigin = it->second;
            opOrigin = joinOrigin(opOrigin, operandOrigin);
          }
        }

        // Propagate to all results
        for (Value result : op.getResults()) {
          auto it = origins.find(result);
          int64_t oldVal =
              (it != origins.end()) ? it->second : kUninitialized;
          int64_t newVal = joinOrigin(oldVal, opOrigin);
          if (newVal != oldVal) {
            origins[result] = newVal;
            changed = true;
          }
        }
      }
    }
  }

  return origins;
}

// Query the DSL arg origin for a Value from the precomputed map.
// Returns the DSL arg index, or -1 if unknown/conflict.
static int64_t queryOrigin(const DenseMap<Value, int64_t> &origins, Value val) {
  auto it = origins.find(val);
  if (it == origins.end())
    return -1;
  return (it->second >= 0) ? it->second : -1;
}

// Analyze the LLVM function's return to determine per-result alias info.
// Returns a vector of DSL arg indices (one per DSLRegionOp result).
// -1 means unknown / no alias.
static SmallVector<int64_t>
analyzeFuncReturnAliases(LLVM::LLVMFuncOp func, size_t numResults,
                         ArrayRef<int64_t> funcArgToDslArg) {
  if (numResults == 0)
    return {};

  // Run forward origin propagation
  DenseMap<Value, int64_t> origins =
      computeDslArgOrigins(func, funcArgToDslArg);

  // Find the return op
  LLVM::ReturnOp retOp = nullptr;
  func.walk([&](LLVM::ReturnOp op) { retOp = op; });
  if (!retOp || retOp.getNumOperands() == 0)
    return SmallVector<int64_t>(numResults, -1);

  Value retVal = retOp.getOperand(0);


  // Single result: query the return value directly
  if (numResults == 1) {
    int64_t origin = queryOrigin(origins, retVal);
    return {origin};
  }

  // Multiple results: outer struct, trace each top-level field independently.
  // Walk the insertvalue chain to find each field's value.
  SmallVector<int64_t> result(numResults, -1);
  Value current = retVal;
  while (auto ivOp = current.getDefiningOp<LLVM::InsertValueOp>()) {
    ArrayRef<int64_t> position = ivOp.getPosition();
    if (position.size() == 1) {
      int64_t fieldIdx = position[0];
      if (fieldIdx >= 0 && fieldIdx < (int64_t)numResults)
        result[fieldIdx] = queryOrigin(origins, ivOp.getValue());
    }
    current = ivOp.getContainer();
  }

  return result;
}

// Compute funcArgToDslArg mapping from DSL arg types.
// Each DSL arg expands to one or more LLVM func args based on its type:
//   RankedTensorType(rank=r) / MemDescType(rank=r) → 3 + 2*r func args
//   PointerType / IntegerType / FloatType → 1 func arg
static SmallVector<int64_t>
computeFuncArgToDslArg(const std::vector<Value> &args) {
  SmallVector<int64_t> mapping;
  int64_t dslArgIdx = 0;
  for (const Value &arg : args) {
    Type ty = arg.getType();
    size_t numFuncArgs = 1;
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
      numFuncArgs = 3 + 2 * tensorTy.getRank();
    else if (auto memdescTy = dyn_cast<ttg::MemDescType>(ty))
      numFuncArgs = 3 + 2 * memdescTy.getShape().size();
    for (size_t i = 0; i < numFuncArgs; ++i)
      mapping.push_back(dslArgIdx);
    dslArgIdx++;
  }
  return mapping;
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
tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
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

  // Compute funcArgToDslArg mapping early for return analysis and later reuse.
  SmallVector<int64_t> funcArgToDslArg = computeFuncArgToDslArg(args);

  // Infer output types by analyzing the return op and tracing return values
  // back to DSL args. This correctly handles:
  //   - void return → no outputs
  //   - scalar return → use LLVM type directly
  //   - single memref descriptor struct return → trace to DSL arg, use its type
  //   - multi-result struct return → count nested structs, trace each to DSL arg
  SmallVector<Type> outputTys;
  Type retTy = funcOp.getFunctionType().getReturnType();
  if (!isa<LLVM::LLVMVoidType>(retTy)) {
    size_t numResults = 1;
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(retTy)) {
      // Count nested struct fields to determine number of results.
      // A single memref descriptor has no nested structs (ptr, ptr, i64,
      // array, array). A multi-result outer struct has nested struct fields.
      size_t structCount = 0;
      for (Type fieldTy : structTy.getBody()) {
        if (isa<LLVM::LLVMStructType>(fieldTy))
          structCount++;
      }
      if (structCount > 0)
        numResults = structCount;
    }
    SmallVector<int64_t> aliases =
        analyzeFuncReturnAliases(func, numResults, funcArgToDslArg);
    for (size_t i = 0; i < numResults; ++i) {
      int64_t idx = aliases[i];
      if (idx >= 0 && idx < (int64_t)args.size()) {
        outputTys.push_back(args[idx].getType());
      } else {
        // Fallback for scalar returns or when tracing fails
        outputTys.push_back(retTy);
      }
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
  // funcArgToDslArg already computed above, reuse it.
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
  // Analyze alias: which DSLRegionOp result aliases which DSL operand
  size_t numResults = dslRegionOp.getNumResults();
  SmallVector<int64_t> aliasOperandIndices =
      analyzeFuncReturnAliases(func, numResults, funcArgToDslArg);
  dslRegionOp->setAttr(
      "tle.alias_operand_indices",
      DenseI64ArrayAttr::get(builder.getContext(), aliasOperandIndices));

  return dslRegionOp;
}