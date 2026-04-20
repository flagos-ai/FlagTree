#include "tle/utils/include/analyzereturn.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton::tle::dataanalyze {

/// Returns true if this OriginSet changed.
bool OriginSet::merge(const OriginSet &other) {
  bool changed = false;
  if (other.conflict && !conflict) {
    conflict = true;
    changed = true;
  }
  for (int64_t idx : other.indices) {
    if (indices.insert(idx))
      changed = true;
  }
  return changed;
}

int64_t getDslArgIdx(BlockArgument blockArg,
                     ArrayRef<int64_t> funcArgToDslArg) {
  int64_t funcIdx = blockArg.getArgNumber();
  if (funcIdx >= 0 && funcIdx < (int64_t)funcArgToDslArg.size())
    return funcArgToDslArg[funcIdx];
  return -1;
}

DenseMap<Value, OriginSet>
computeDslArgOrigins(LLVM::LLVMFuncOp func, ArrayRef<int64_t> funcArgToDslArg) {
  DenseMap<Value, OriginSet> origins;

  // Initialize all block arguments with their DSL arg origin
  for (Block &block : func.getBlocks()) {
    for (BlockArgument arg : block.getArguments()) {
      int64_t dslIdx = getDslArgIdx(arg, funcArgToDslArg);
      OriginSet &os = origins[arg];
      if (dslIdx >= 0)
        os.indices.insert(dslIdx);
      else
        os.conflict = true; // block arg not mapped to any DSL arg
    }
  }

  // Iterate until fixpoint
  bool changed = true;
  while (changed) {
    changed = false;
    for (Block &block : func.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        // undef/poison: empty set (neutral element), leave as default
        if (isa<LLVM::UndefOp, LLVM::PoisonOp>(op)) {
          for (Value result : op.getResults()) {
            origins.try_emplace(result);
          }
          continue;
        }

        // Compute merged origin of all operands
        OriginSet opOrigin;
        if (op.getNumOperands() == 0) {
          opOrigin.conflict = true;
        } else {
          for (Value operand : op.getOperands()) {
            auto it = origins.find(operand);
            if (it != origins.end())
              opOrigin.merge(it->second);
          }
        }

        // Propagate to all results
        for (Value result : op.getResults()) {
          OriginSet &resultOrigin = origins[result];
          if (resultOrigin.merge(opOrigin))
            changed = true;
        }
      }
    }
  }

  return origins;
}

SmallVector<int64_t>
analyzeFuncReturnAliases(LLVM::LLVMFuncOp func,
                         ArrayRef<int64_t> funcArgToDslArg) {
  DenseMap<Value, OriginSet> origins =
      computeDslArgOrigins(func, funcArgToDslArg);

  LLVM::ReturnOp retOp = nullptr;
  func.walk([&](LLVM::ReturnOp op) { retOp = op; });
  if (!retOp || retOp.getNumOperands() == 0)
    return {};

  Value retVal = retOp.getOperand(0);
  auto it = origins.find(retVal);
  if (it == origins.end() || it->second.conflict || it->second.indices.empty())
    func.emitError("return value cannot be traced back to any DSL argument");

  return SmallVector<int64_t>(it->second.indices.begin(),
                              it->second.indices.end());
}

SmallVector<int64_t> computeFuncArgToDslArg(const std::vector<Value> &args) {
  SmallVector<int64_t> mapping;
  int64_t dslArgIdx = 0;
  for (const Value &arg : args) {
    Type ty = arg.getType();
    size_t numFuncArgs = 1;
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
      numFuncArgs = 3 + 2 * tensorTy.getRank();
    else if (auto memdescTy = dyn_cast<mlir::triton::gpu::MemDescType>(ty))
      numFuncArgs = 3 + 2 * memdescTy.getShape().size();
    for (size_t i = 0; i < numFuncArgs; ++i)
      mapping.push_back(dslArgIdx);
    dslArgIdx++;
  }
  return mapping;
}

} // namespace mlir::triton::tle::dataanalyze
