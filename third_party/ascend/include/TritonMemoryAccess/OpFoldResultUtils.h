/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_MEMORY_ACCESS_OP_FOLD_RESULT_UTILS_H
#define TRITON_MEMORY_ACCESS_OP_FOLD_RESULT_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir {

class OpBuilder;

enum class IntegerExtensionKind {
  Signed,
  Unsigned,
};

FailureOr<Value>
castIntegerLike(OpBuilder &builder, Location loc, Value value, Type targetType,
                IntegerExtensionKind extension = IntegerExtensionKind::Signed);

std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg);

/// Without an explicit result type, preserve equal types and widen signless
/// integers. Mixed index/integer operands require an explicit result type.
OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder,
                             Type resultType = {});

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder);

/// Uses the same result-type rules as addOpFoldResult.
OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder,
                             Type resultType = {});

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder);

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder);

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder);

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &builder);

std::optional<int64_t> getIntAttr(const OpFoldResult ofr);

Value materializeValue(OpBuilder &builder, Location loc, OpFoldResult ofr);

bool isZero(const OpFoldResult ofr);

bool isOne(const OpFoldResult ofr);

Value convertToIndexIfNeeded(Value input, const Location &loc,
                             OpBuilder &builder);

RankedTensorType getExtractSlicedType(llvm::ArrayRef<OpFoldResult> shape,
                                      const llvm::SmallBitVector &droppedDims,
                                      Type elemType);

OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder);

} // namespace mlir

#endif // TRITON_MEMORY_ACCESS_OP_FOLD_RESULT_UTILS_H
