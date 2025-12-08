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
#include <limits>

#include "Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gcu {

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (llvm::isa<Attribute>(ofr) && isa<IntegerAttr>(ofr.get<Attribute>()))
    return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();

  return std::nullopt;
}

Value getValue(OpBuilder &builder, Location loc, const OpFoldResult ofr) {
  if (auto attr = getIntAttr(ofr)) {
    return builder.create<arith::ConstantIndexOp>(loc, attr.value())
        .getResult();
  } else {
    return dyn_cast<Value>(ofr);
  }
}

llvm::SmallVector<Value> getValues(OpBuilder &builder, Location loc,
                                   const llvm::SmallVector<OpFoldResult> &ofr) {
  llvm::SmallVector<Value> values;
  for (uint64_t i = 0; i < ofr.size(); ++i) {
    values.push_back(getValue(builder, loc, ofr[i]));
  }
  return values;
}

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat and sitofp) that produce it and to extract they
// underlying scalar value . If no scalar value can be extracted, a nullptr is
// returned.
std::optional<Value> getScalarValue(OpBuilder &builder, Location loc, Value v) {
  // Record if an sitofp op was in the chain of ops that produce the scalar
  Operation *siToFp = nullptr;

  while (true) {
    if (!dyn_cast<ShapedType>(v.getType())) {
      break;
    }

    if (auto op = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }

        auto typedAttr = attr.getSplatValue<TypedAttr>();
        v = builder.create<arith::ConstantOp>(loc, attr.getElementType(),
                                              typedAttr);
      }
    } else if (auto op = v.getDefiningOp<triton::SplatOp>()) {
      v = op.getSrc();
    } else if (auto op = v.getDefiningOp<arith::SIToFPOp>()) {
      siToFp = op;
      v = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }

  if (siToFp) {
    auto resType = siToFp->getResult(0).getType();
    if (auto shapedType = dyn_cast<ShapedType>(resType)) {
      resType = shapedType.getElementType();
    }
    return builder.create<arith::SIToFPOp>(loc, resType, v);
  }

  return v;
}

OpFoldResult addOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  } else {
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  } else {
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return builder.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = builder.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRValue(OpBuilder &builder, Location loc,
                         const OpFoldResult lhs, const Value rhs) {
  auto lhsIntAttr = getIntAttr(lhs);

  auto rhsIsConst = false;
  // if rhs is not a const, use max value since min is used to represent
  // dynamic size or stride
  auto rhsConstValue = std::numeric_limits<int64_t>::max();
  auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
  if (rhsOp) {
    rhsIsConst = true;
    rhsConstValue = cast<IntegerAttr>(rhsOp.getValue()).getInt();
  }

  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIsConst) {
    if (rhsConstValue == 0)
      return rhsOp.getResult();
    if (rhsConstValue == 1)
      return lhs;
  }

  // 0. both lhs and rhs are constants
  if (lhsIntAttr && rhsIsConst)
    return builder.getIndexAttr(lhsIntAttr.value() * rhsConstValue);

  // 1. if lhs is constant but rhs is not
  if (lhsIntAttr && !rhsIsConst) {
    auto lhsConstOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    auto mulOp =
        builder.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
    return mulOp.getResult();
  }

  // 2. if lhs is not constant
  assert(!lhsIntAttr);
  auto mulOp = builder.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs);
  return mulOp.getResult();
}

OpFoldResult minOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(
        std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = builder.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

OpFoldResult maxOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(
        std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto maxOp = builder.create<arith::MaxSIOp>(loc, lhsValue, rhsValue);
  return maxOp.getResult();
}

OpFoldResult remOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() % rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto remOp = builder.create<arith::RemSIOp>(loc, lhsValue, rhsValue);
  return remOp.getResult();
}

OpFoldResult divOFRs(OpBuilder &builder, Location loc, const OpFoldResult lhs,
                     const OpFoldResult rhs) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() / rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto divOp = builder.create<arith::DivSIOp>(loc, lhsValue, rhsValue);
  return divOp.getResult();
}

} // namespace gcu
} // namespace triton
} // namespace mlir
