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

namespace mlir::triton::utils {

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
} // namespace mlir::triton::utils

#endif // ZTC_CONVERSION_UTILS_H
