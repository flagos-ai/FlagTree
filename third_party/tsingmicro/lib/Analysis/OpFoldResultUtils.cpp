//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (isa<Attribute>(ofr) && isa<IntegerAttr>(cast<Attribute>(ofr)))
    return dyn_cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();

  return std::nullopt;
}

bool hasConstZero(const OpFoldResult ofr) {
  auto intAttr = getIntAttr(ofr);
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  auto val = dyn_cast<Value>(ofr);
  assert(val);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  intAttr = getIntAttr(constOp.getValue());
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  return false;
}

Value ofrToIndexValue(const OpFoldResult ofr, const Location loc,
                      OpBuilder &b) {
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = b.create<arith::IndexCastOp>(loc, b.getIndexType(), val);
    }
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b) {
  return llvm::to_vector<4>(
      llvm::map_range(ofrs, [&](OpFoldResult ofr) -> Value {
        return ofrToIndexValue(ofr, loc, b);
      }));
}

OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  } else {
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  } else {
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = b.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc, OpBuilder &b) {
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
    return b.getIndexAttr(lhsIntAttr.value() * rhsConstValue);

  // 1. if lhs is constant but rhs is not
  if (lhsIntAttr && !rhsIsConst) {
    auto lhsConstOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    auto mulOp = b.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
    return mulOp.getResult();
  }

  // 2. if lhs is not constant
  assert(!lhsIntAttr);
  auto mulOp = b.create<arith::MulIOp>(loc, cast<Value>(lhs), rhs);
  return mulOp.getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = b.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto maxOp = b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue);
  return maxOp.getResult();
}

OpFoldResult compareOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                         const arith::CmpIPredicate pred,
                         const OpFoldResult trueOFR,
                         const OpFoldResult falseOFR, const Location loc,
                         OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return the result directly
  if (lhsIntAttr && rhsIntAttr) {
    switch (pred) {
    case arith::CmpIPredicate::eq:
      return *lhsIntAttr == *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::ne:
      return *lhsIntAttr != *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return *lhsIntAttr < *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return *lhsIntAttr <= *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return *lhsIntAttr > *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return *lhsIntAttr >= *rhsIntAttr ? trueOFR : falseOFR;
    default:
      llvm_unreachable("Unsupported predicate");
    }
  }

  auto lhsValue = ofrToIndexValue(lhs, loc, b);
  auto rhsValue = ofrToIndexValue(rhs, loc, b);
  auto trueValue = ofrToIndexValue(trueOFR, loc, b);
  auto falseValue = ofrToIndexValue(falseOFR, loc, b);

  auto cmpOp = b.create<arith::CmpIOp>(loc, pred, lhsValue, rhsValue);
  auto selectOp = b.create<arith::SelectOp>(loc, cmpOp, trueValue, falseValue);
  return selectOp.getResult();
}
} // namespace mlir
