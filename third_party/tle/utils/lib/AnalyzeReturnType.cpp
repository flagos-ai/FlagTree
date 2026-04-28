#include "tle/utils/include/AnalyzeReturnType.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton::tle::data_analyze {

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
  if (funcIdx >= 0 && funcIdx < static_cast<int64_t>(funcArgToDslArg.size()))
    return funcArgToDslArg[funcIdx];
  return -1;
}

DenseMap<Value, OriginSet>
computeDslArgOrigins(LLVM::LLVMFuncOp func, ArrayRef<int64_t> funcArgToDslArg) {
  DenseMap<Value, OriginSet> origins;

  // Initialize only entry block arguments from funcArgToDslArg.
  // Non-entry block arguments get their origins via CFG propagation below.
  Block &entryBlock = func.getBody().front();
  for (BlockArgument arg : entryBlock.getArguments()) {
    int64_t dslIdx = getDslArgIdx(arg, funcArgToDslArg);
    OriginSet &os = origins[arg];
    if (dslIdx >= 0)
      os.indices.insert(dslIdx);
    else
      os.conflict = true;
  }

  // Pre-insert non-entry block arguments into the map so that later
  // insertions during the fixpoint loop do not invalidate iterators.
  for (Block &block : func.getBlocks()) {
    if (&block == &entryBlock)
      continue;
    for (BlockArgument arg : block.getArguments())
      origins.try_emplace(arg);
  }

  // Iterate until fixpoint
  bool changed = true;
  while (changed) {
    changed = false;

    // --- CFG propagation via BranchOpInterface: propagate origins from
    //     branch operands to successor block arguments (phi edges). ---
    for (Block &block : func.getBlocks()) {
      if (&block == &entryBlock)
        continue;
      for (auto it = block.pred_begin(), e = block.pred_end(); it != e; ++it) {
        Block *pred = *it;
        Operation *term = pred->getTerminator();
        if (auto branchOp = dyn_cast<BranchOpInterface>(term)) {
          unsigned succIdx = it.getSuccessorIndex();
          SuccessorOperands succOperands =
              branchOp.getSuccessorOperands(succIdx);
          for (auto [idx, arg] : llvm::enumerate(block.getArguments())) {
            Value operand = succOperands[idx];
            if (!operand)
              continue;
            auto opIt = origins.find(operand);
            if (opIt != origins.end()) {
              OriginSet srcOrigin = opIt->second;
              auto argIt = origins.find(arg);
              if (argIt->second.merge(srcOrigin))
                changed = true;
            }
          }
        }
      }
    }

    // --- Intra-block propagation: propagate origins through operations. ---
    for (Block &block : func.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        // undef/poison: empty set (neutral element), leave as default
        if (isa<LLVM::UndefOp, LLVM::PoisonOp>(op)) {
          for (Value result : op.getResults()) {
            origins.try_emplace(result);
          }
        } else {
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
            auto [resIt, _] = origins.try_emplace(result);
            if (resIt->second.merge(opOrigin))
              changed = true;
          }
        }
      }
    }
  }

  return origins;
}

FailureOr<SmallVector<int64_t>>
analyzeFuncReturnAliases(LLVM::LLVMFuncOp func,
                         ArrayRef<int64_t> funcArgToDslArg) {
  DenseMap<Value, OriginSet> origins =
      computeDslArgOrigins(func, funcArgToDslArg);

  LLVM::ReturnOp retOp = nullptr;
  func.walk([&](LLVM::ReturnOp op) { retOp = op; });

  Value retVal = retOp.getOperand(0);
  auto it = origins.find(retVal);
  if (it == origins.end() || it->second.conflict || it->second.indices.empty())
    return func.emitError(
        "return value cannot be traced back to any DSL argument");

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

} // namespace mlir::triton::tle::data_analyze
