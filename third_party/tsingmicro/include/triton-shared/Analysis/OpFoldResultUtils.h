//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_OPFOLDRESULT_UTILS_H
#define TRITON_ANALYSIS_OPFOLDRESULT_UTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <optional>

namespace mlir {

class OpBuilder;

// Return integer if ofr is an IntegerAttr. Note that this function differs
// from getConstantIntValue, which returns an integer if ofr is the constant
// result of an operation too.
std::optional<int64_t> getIntAttr(const OpFoldResult ofr);

// Return if ofr contains a constant zero, either represented by an integer
// attribute or a constant value.
bool hasConstZero(const OpFoldResult ofr);

// Create a value of index type if necessary from an OpFoldResult.
Value ofrToIndexValue(const OpFoldResult ofr, const Location loc, OpBuilder &b);

// Create a vector of values of index type if necessary from an array of
// OpFoldResults.
SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b);

// Process addition of two OFRs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

// Process multiplication of two OFRs. If both OFRs are Integer Attributes,
// result is an Integer Attribtue. Otherwise, insert the arith.muli
// instruction if needed and use its result Value.
OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc, OpBuilder &b);

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b);

OpFoldResult compareOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                    const arith::CmpIPredicate pred, const OpFoldResult trueVal,
                    const OpFoldResult falseVal, const Location loc, OpBuilder &b);
} // namespace mlir

#endif
