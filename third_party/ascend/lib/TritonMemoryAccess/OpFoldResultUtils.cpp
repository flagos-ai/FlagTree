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

#include "TritonMemoryAccess/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <optional>

namespace mlir {
namespace {

Value createConstIndexValueOp(const Location &loc, OpBuilder &builder,
                              int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value))
      .getResult();
}

FailureOr<Value> materializeIndexOperand(const OpFoldResult &operand,
                                         std::optional<int64_t> constant,
                                         Location loc, OpBuilder &builder) {
  if (constant)
    return createConstIndexValueOp(loc, builder, *constant);
  auto value = dyn_cast<Value>(operand);
  if (!value)
    return failure();
  return castIntegerLike(builder, loc, value, builder.getIndexType());
}

Type getOpFoldResultType(const OpFoldResult &operand) {
  if (auto value = dyn_cast<Value>(operand))
    return value.getType();
  if (auto attribute = dyn_cast<Attribute>(operand)) {
    if (auto typedAttribute = dyn_cast<TypedAttr>(attribute))
      return typedAttribute.getType();
  }
  return Type();
}

bool isSupportedScalarIntegerType(Type type) {
  return type && (type.isIndex() || type.isSignlessInteger());
}

FailureOr<Type> resolveIntegerResultType(const OpFoldResult &lhs,
                                         const OpFoldResult &rhs,
                                         Type requestedType) {
  Type lhsType = getOpFoldResultType(lhs);
  Type rhsType = getOpFoldResultType(rhs);
  if (!isSupportedScalarIntegerType(lhsType) ||
      !isSupportedScalarIntegerType(rhsType) ||
      (requestedType && !isSupportedScalarIntegerType(requestedType)))
    return failure();

  if (requestedType)
    return requestedType;
  if (lhsType == rhsType)
    return lhsType;

  // Index has no fixed IR-level width, so it does not participate in default
  // widening against ordinary integers.
  auto lhsInteger = dyn_cast<IntegerType>(lhsType);
  auto rhsInteger = dyn_cast<IntegerType>(rhsType);
  if (!lhsInteger || !rhsInteger ||
      lhsInteger.getWidth() == rhsInteger.getWidth())
    return failure();
  return lhsInteger.getWidth() > rhsInteger.getWidth() ? lhsType : rhsType;
}

FailureOr<OpFoldResult> castIntegerFoldResult(const OpFoldResult &operand,
                                              Type targetType, Location loc,
                                              OpBuilder &builder) {
  if (std::optional<int64_t> constant = getConstantOfAttr(operand))
    return OpFoldResult(builder.getIntegerAttr(targetType, *constant));

  auto value = dyn_cast<Value>(operand);
  if (!value)
    return failure();
  FailureOr<Value> converted = castIntegerLike(builder, loc, value, targetType);
  if (failed(converted))
    return failure();
  return OpFoldResult(*converted);
}

FailureOr<Value> materializeIntegerOperand(const OpFoldResult &operand,
                                           std::optional<int64_t> constant,
                                           Type targetType, Location loc,
                                           OpBuilder &builder) {
  if (constant)
    return builder
        .create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(targetType, *constant))
        .getResult();
  auto value = dyn_cast<Value>(operand);
  if (!value)
    return failure();
  return castIntegerLike(builder, loc, value, targetType);
}

} // namespace

std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg) {
  if (isa<Attribute>(arg))
    return getConstantIntValue(arg);
  return std::nullopt;
}

FailureOr<Value> castIntegerLike(OpBuilder &builder, Location loc, Value value,
                                 Type targetType,
                                 IntegerExtensionKind extension) {
  if (!value || !targetType)
    return failure();

  Type sourceType = value.getType();
  if (sourceType == targetType)
    return value;

  if ((sourceType.isIndex() && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(sourceType) && targetType.isIndex())) {
    if (extension == IntegerExtensionKind::Unsigned)
      return builder.create<arith::IndexCastUIOp>(loc, targetType, value)
          .getResult();
    return builder.create<arith::IndexCastOp>(loc, targetType, value)
        .getResult();
  }

  auto extendOrTruncate = [&](unsigned sourceWidth,
                              unsigned targetWidth) -> FailureOr<Value> {
    if (sourceWidth < targetWidth) {
      if (extension == IntegerExtensionKind::Unsigned)
        return builder.create<arith::ExtUIOp>(loc, targetType, value)
            .getResult();
      return builder.create<arith::ExtSIOp>(loc, targetType, value).getResult();
    }
    if (sourceWidth > targetWidth)
      return builder.create<arith::TruncIOp>(loc, targetType, value)
          .getResult();
    return failure();
  };

  auto sourceInteger = dyn_cast<IntegerType>(sourceType);
  auto targetInteger = dyn_cast<IntegerType>(targetType);
  if (sourceInteger && targetInteger)
    return extendOrTruncate(sourceInteger.getWidth(), targetInteger.getWidth());

  auto sourceTensor = dyn_cast<RankedTensorType>(sourceType);
  auto targetTensor = dyn_cast<RankedTensorType>(targetType);
  if (!sourceTensor || !targetTensor ||
      sourceTensor.getShape() != targetTensor.getShape() ||
      sourceTensor.getEncoding() != targetTensor.getEncoding())
    return failure();

  auto sourceElement = dyn_cast<IntegerType>(sourceTensor.getElementType());
  auto targetElement = dyn_cast<IntegerType>(targetTensor.getElementType());
  if (!sourceElement || !targetElement)
    return failure();

  return extendOrTruncate(sourceElement.getWidth(), targetElement.getWidth());
}

OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b,
                             Type resultType) {
  FailureOr<Type> resolvedType = resolveIntegerResultType(lhs, rhs, resultType);
  if (failed(resolvedType))
    return OpFoldResult();

  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIntegerAttr(*resolvedType, *lhsInt + *rhsInt);

  if (!lhsInt && rhsInt && *rhsInt == 0) {
    FailureOr<OpFoldResult> result =
        castIntegerFoldResult(lhs, *resolvedType, loc, b);
    return succeeded(result) ? *result : OpFoldResult();
  }
  if (!rhsInt && lhsInt && *lhsInt == 0) {
    FailureOr<OpFoldResult> result =
        castIntegerFoldResult(rhs, *resolvedType, loc, b);
    return succeeded(result) ? *result : OpFoldResult();
  }

  FailureOr<Value> lhsValue =
      materializeIntegerOperand(lhs, lhsInt, *resolvedType, loc, b);
  FailureOr<Value> rhsValue =
      materializeIntegerOperand(rhs, rhsInt, *resolvedType, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::AddIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() - rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;

  FailureOr<Value> lhsValue = materializeIndexOperand(lhs, lhsInt, loc, b);
  FailureOr<Value> rhsValue = materializeIndexOperand(rhs, rhsInt, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::SubIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b,
                             Type resultType) {
  FailureOr<Type> resolvedType = resolveIntegerResultType(lhs, rhs, resultType);
  if (failed(resolvedType))
    return OpFoldResult();

  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIntegerAttr(*resolvedType, *lhsInt * *rhsInt);

  if (lhsInt) {
    if (*lhsInt == 0)
      return b.getIntegerAttr(*resolvedType, 0);
    if (*lhsInt == 1) {
      FailureOr<OpFoldResult> result =
          castIntegerFoldResult(rhs, *resolvedType, loc, b);
      return succeeded(result) ? *result : OpFoldResult();
    }
  }
  if (rhsInt) {
    if (*rhsInt == 0)
      return b.getIntegerAttr(*resolvedType, 0);
    if (*rhsInt == 1) {
      FailureOr<OpFoldResult> result =
          castIntegerFoldResult(lhs, *resolvedType, loc, b);
      return succeeded(result) ? *result : OpFoldResult();
    }
  }

  FailureOr<Value> lhsValue =
      materializeIntegerOperand(lhs, lhsInt, *resolvedType, loc, b);
  FailureOr<Value> rhsValue =
      materializeIntegerOperand(rhs, rhsInt, *resolvedType, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::MulIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot div 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() / rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  if (rhsInt) {
    if (rhsInt.value() == 1)
      return lhs;
  }

  FailureOr<Value> lhsValue = materializeIndexOperand(lhs, lhsInt, loc, b);
  FailureOr<Value> rhsValue = materializeIndexOperand(rhs, rhsInt, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::DivSIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot remainder by 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() % rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  FailureOr<Value> lhsValue = materializeIndexOperand(lhs, lhsInt, loc, b);
  FailureOr<Value> rhsValue = materializeIndexOperand(rhs, rhsInt, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::RemSIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::min(lhsInt.value(), rhsInt.value()));

  FailureOr<Value> lhsValue = materializeIndexOperand(lhs, lhsInt, loc, b);
  FailureOr<Value> rhsValue = materializeIndexOperand(rhs, rhsInt, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::MinSIOp>(loc, *lhsValue, *rhsValue).getResult();
}

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::max(lhsInt.value(), rhsInt.value()));

  FailureOr<Value> lhsValue = materializeIndexOperand(lhs, lhsInt, loc, b);
  FailureOr<Value> rhsValue = materializeIndexOperand(rhs, rhsInt, loc, b);
  if (failed(lhsValue) || failed(rhsValue))
    return OpFoldResult();

  return b.create<arith::MaxSIOp>(loc, *lhsValue, *rhsValue).getResult();
}
std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  Attribute attr;
  if (auto val = dyn_cast<Value>(ofr)) {
    if (!val.getDefiningOp<arith::ConstantOp>())
      return std::nullopt;
    attr = cast<IntegerAttr>(val.getDefiningOp<arith::ConstantOp>().getValue());
  } else {
    attr = dyn_cast<Attribute>(ofr);
  }
  if (attr && isa<IntegerAttr>(attr))
    return dyn_cast<IntegerAttr>(attr).getInt();
  return std::nullopt;
}

Value materializeValue(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>()) {
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(intVal.value()));
  }
  assert(intVal.has_value());
  return Value();

  // return builder.create<arith::ConstantIndexOp>(
  //     loc, dyn_cast<IntegerAttr>(attr).getInt());
}

bool isZero(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 0;
}

bool isOne(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 1;
}

Value convertToIndexIfNeeded(Value input, const Location &loc, OpBuilder &b) {
  auto inputType = input.getType();
  if (auto intType = dyn_cast<IntegerType>(inputType)) {
    if (intType.isInteger(32) || intType.isInteger(64)) {
      return b.create<arith::IndexCastOp>(loc, b.getIndexType(), input);
    }
  }
  return input;
}

RankedTensorType getExtractSlicedType(ArrayRef<OpFoldResult> shape,
                                      const llvm::SmallBitVector &droppedDims,
                                      Type elemType) {
  SmallVector<int64_t> targetShape;
  for (auto [idx, dimOfr] : llvm::enumerate(shape)) {
    if (!droppedDims[idx]) {
      if (auto dim = getConstantIntValue(dimOfr)) {
        targetShape.push_back(dim.value());
      } else {
        targetShape.push_back(ShapedType::kDynamic);
      }
    }
  }
  return RankedTensorType::get(targetShape, elemType);
}

// Fold layout constant info to attr, otherwise convert to index type value.
OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder) {
  OpFoldResult constantFold = getAsOpFoldResult(value);
  if (llvm::isa<Attribute>(constantFold)) {
    assert(isa<IntegerAttr>(cast<Attribute>(constantFold)));
    return constantFold;
  }

  Type sourceType = value.getType();
  if (!sourceType.isIndex() && !isa<IntegerType>(sourceType))
    llvm_unreachable("Illegal data type when parse block data layout info");

  IntegerExtensionKind extension = sourceType.isInteger(/*width=*/1)
                                       ? IntegerExtensionKind::Unsigned
                                       : IntegerExtensionKind::Signed;
  FailureOr<Value> converted = castIntegerLike(
      builder, value.getLoc(), value, builder.getIndexType(), extension);
  if (failed(converted))
    llvm_unreachable("Failed to convert block data layout info to index");
  return *converted;
}

} // namespace mlir
