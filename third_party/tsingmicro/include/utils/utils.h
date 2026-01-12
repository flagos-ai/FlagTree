//===------------------- utils.h ------------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Utility functions for ztc conversion.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_UTILS_H
#define ZTC_CONVERSION_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h" // Include the header for Type

using namespace mlir;

namespace mlir::triton {

// Check if the type is a pointer type or a tensor of pointers
bool isPtrTypeLike(Type t);

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder);

Value declareTx81Function(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes);

bool isOperandMemorySpaceSPM(Value operand);

// NOTE: Reduction can lower to target reduction ops or target elementwise ops.

// Target support reduce instruction
// Different target ops have different supported types restriction
bool isTypeRestrictedTargetSupportedReductionOp(mlir::Operation *redOp);
// Integer and float type reduction op
bool isTargetSupportedReductionOp(mlir::Operation *redOp);

// Reduce to elementwise op
bool isTypeRestrictedTargetSupportedReduceToElementWiseOp(
    mlir::Operation *redOp);
// Integer and float type reduction op
bool isTargetSupportedReduceToElementWiseOp(mlir::Operation *redOp);

bool isTritonAllowedReductionOp(Operation *redOp);

// Reduction ops and elementwise op only support float types.
bool isTargetSupportedFloatType(Type elementType);
bool isTargetSupportedType(Type elementType);

bool isReductionOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                           Type elementType);
bool isReduceToElementWiseOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                                     Type elementType,
                                                     int64_t elemCount,
                                                     int64_t rank);

template <typename T> llvm::SmallVector<Operation *> getRegionOps(T linalgOp) {
  assert((linalgOp->getNumRegions() == 1 &&
          linalgOp->getRegion(0).hasOneBlock()) &&
         "It only applies to ops with one region and one block!");
  auto regionBlock = linalgOp.getBlock();
  return llvm::map_to_vector(regionBlock->without_terminator(),
                             [](Operation &op) { return &op; });
}
} // namespace mlir::triton

// Gelu mode.
enum class GeluMode { None = 0, Tanh = 1 };

#endif // ZTC_CONVERSION_UTILS_H
