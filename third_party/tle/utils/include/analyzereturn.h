#ifndef TLE_UTILS_ANALYZERETURN_H
#define TLE_UTILS_ANALYZERETURN_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::tle::dataanalyze {

/// Origin state for a Value during propagation.
/// Each Value maps to a set of DSL arg indices it originates from.
/// A special "conflict" flag indicates the value has non-DSL-arg origins
/// (e.g., constants), making it unreachable for return type inference.
struct OriginSet {
  llvm::SetVector<int64_t> indices; // DSL arg indices this value comes from
  bool conflict = false;            // true if any non-DSL-arg origin is mixed in

  /// Merge another OriginSet into this one (union of indices).
  /// If either side is conflict, the result is conflict.
  bool merge(const OriginSet &other);

  /// Check if this origin set has changed compared to another.
  bool operator!=(const OriginSet &other) const;
};

/// Helper: get the DSL arg index for a BlockArgument via funcArgToDslArg map.
/// Returns -1 if out of range.
int64_t getDslArgIdx(mlir::BlockArgument blockArg,
                     llvm::ArrayRef<int64_t> funcArgToDslArg);

/// Run forward origin propagation on a function using set-based (meet) semantics.
/// Each Value's origin is a set of DSL arg indices.
/// - Block args: initialized from funcArgToDslArg
/// - undef/poison: empty set (neutral element)
/// - Constants (no operands): conflict
/// - Other ops: union of all operand origin sets; conflict if any operand is conflict
llvm::DenseMap<mlir::Value, OriginSet>
computeDslArgOrigins(mlir::LLVM::LLVMFuncOp func,
                     llvm::ArrayRef<int64_t> funcArgToDslArg);

/// Analyze the LLVM function's return to determine per-result alias info.
/// Returns a sorted vector of DSL arg indices that the return value originates from.
/// If the return value has any non-DSL-arg origin (conflict), returns empty vector.
/// numResults = size of the returned vector.
llvm::SmallVector<int64_t>
analyzeFuncReturnAliases(mlir::LLVM::LLVMFuncOp func,
                         llvm::ArrayRef<int64_t> funcArgToDslArg);

/// Compute funcArgToDslArg mapping from DSL arg types.
/// Each DSL arg expands to one or more LLVM func args based on its type:
///   RankedTensorType(rank=r) / MemDescType(rank=r) → 3 + 2*r func args
///   PointerType / IntegerType / FloatType → 1 func arg
llvm::SmallVector<int64_t>
computeFuncArgToDslArg(const std::vector<mlir::Value> &args);

} // namespace mlir::triton::tle::dataanalyze

#endif // TLE_UTILS_ANALYZERETURN_H