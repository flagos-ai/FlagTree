/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCU_ANALYSIS_OPFOLDRESULTUTILS_H
#define GCU_ANALYSIS_OPFOLDRESULTUTILS_H

#include <optional>

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

//===--------------------------------------------------------------------===//
// The main code is modified from triton-to-linalg branch in triton repo.
//===--------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gcu {

// Return integer if ofr is an IntegerAttr. Note that this function differs
// from getConstantIntValue, which returns an integer if ofr is the constant
// result of an operation too.
std::optional<int64_t> getIntAttr(const OpFoldResult ofr);

Value getValue(OpBuilder &builder, Location loc, const OpFoldResult ofr);

llvm::SmallVector<Value> getValues(OpBuilder &builder, Location loc,
                                   const llvm::SmallVector<OpFoldResult> &ofr);

std::optional<Value> getScalarValue(OpBuilder &builder, Location loc, Value v);

// Process addition of two OFRs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult addOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
// needed and use its result Value.
OpFoldResult subOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

// Process multiplication of two OFRs. If both OFRs are Integer Attributes,
// result is an Integer Attribute. Otherwise, insert the arith.muli
// instruction if needed and use its result Value.
OpFoldResult mulOFRValue(OpBuilder &builder, Location loc,
                         const OpFoldResult lhs, const Value rhs);

OpFoldResult minOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

OpFoldResult maxOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

// Produce result = lhs % rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.remsi instruction if
// needed and use its result Value.
OpFoldResult remOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

// Produce result = lhs / rhs. If both OFRs are Integer Attributes, result
// is an Integer Attribute. Otherwise, insert the arith.divsi instruction if
// needed and use its result Value.
OpFoldResult divOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs);

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // GCU_ANALYSIS_OPFOLDRESULTUTILS_H
