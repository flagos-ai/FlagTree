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
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "triton-ptr-analysis"

#include "llvm/Support/raw_ostream.h"

class CmpIOpListener : public mlir::OpBuilder::Listener {
public:
  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous) override {
    (void)previous;
    if (auto cmpiOp = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
      auto lhs = cmpiOp.getLhs();
      auto rhs = cmpiOp.getRhs();
      if (lhs.getType() != rhs.getType()) {
        llvm::errs()
            << "[FATAL] CmpIOp created with mismatched operand types:\n";
        cmpiOp->dump();
        llvm::errs() << "LHS type: ";
        lhs.getType().dump();
        llvm::errs() << "RHS type: ";
        rhs.getType().dump();
        llvm_unreachable("CmpIOp with mismatched types created");
      }
    }
  }
};

// Helper to ensure the listener is set on the provided builder (thread-local
// once).
static inline void ensureCmpListener(mlir::OpBuilder &b) {
  static thread_local bool listenerSet = false;
  if (!listenerSet) {
    static CmpIOpListener listener;
    b.setListener(&listener);
    listenerSet = true;
  }
}

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
  ensureCmpListener(b);
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = arith::IndexCastOp::create(b, loc, b.getIndexType(), val);
    }
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  return llvm::to_vector<4>(
      llvm::map_range(ofrs, [&](OpFoldResult ofr) -> Value {
        return ofrToIndexValue(ofr, loc, b);
      }));
}

Value indexTypeCast(Value v, Type targetTy, const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  Type ty = v.getType();
  if (isa<IndexType>(targetTy) || isa<IndexType>(ty)) {
    assert((isa<IntegerType>(targetTy) || isa<IntegerType>(ty)) &&
           "Only cast between index type and integer type");
    return arith::IndexCastOp::create(b, loc, targetTy, v).getResult();
  } else {
    auto targetIntTy = cast<IntegerType>(targetTy);
    auto intTy = cast<IntegerType>(ty);
    if (targetIntTy.getWidth() > intTy.getWidth())
      return arith::ExtSIOp::create(b, loc, targetTy, v).getResult();
    else
      return arith::TruncIOp::create(b, loc, targetTy, v).getResult();
  }
}

OpFoldResult expandOFRIndex(OpFoldResult ofr, OpFoldResult targetForTy,
                            const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  if (getIntAttr(targetForTy))
    return ofr;
  Value targetValueForTy = cast<Value>(targetForTy);
  Type targetTy = targetValueForTy.getType();
  auto targetShapedTy = dyn_cast<ShapedType>(targetTy);

  Value v = dyn_cast<Value>(ofr);
  if (!v)
    v = arith::ConstantOp::create(b, loc,
                                  cast<IntegerAttr>(cast<Attribute>(ofr)));

  Type ty = v.getType();
  if (targetTy == ty)
    return ofr;

  auto shapedTy = dyn_cast<ShapedType>(ty);
  if (targetShapedTy && !shapedTy) {
    Type targetEltTy = targetShapedTy.getElementType();
    // cast to target element type first.
    if (targetEltTy != ty)
      v = indexTypeCast(v, targetEltTy, loc, b);
    return triton::SplatOp::create(b, loc, targetTy, v).getResult();
  } else if (targetShapedTy && shapedTy) {
    Type targetEltTy = targetShapedTy.getElementType();
    Type eltTy = shapedTy.getElementType();
    if (targetShapedTy.getShape() != shapedTy.getShape()) {
      assert(targetEltTy == eltTy &&
             "Only cast between same element type shaped types");
      // This path is for case like:
      // input_ptr + (row_indices[:, None] + row_offsets[:,None] % mod_offset) *
      //   stride_m + col_offsets[None, :] * stride_n
      // The modulo will be in shape of [ROW_SIZE, 1] while row_indices is in
      // shape of [ROW_SIZE,].
      LLVM_DEBUG({
        llvm::dbgs() << "Reshaping ";
        shapedTy.dump();
        llvm::dbgs() << " to ";
        targetShapedTy.dump();
      });
      SmallVector<Value> shapeValues;
      for (auto dim : targetShapedTy.getShape()) {
        shapeValues.push_back(
            arith::ConstantOp::create(b, loc, b.getIndexAttr(dim)));
      }
      RankedTensorType targetShapeTensorTy = RankedTensorType::get(
          targetShapedTy.getShape().size(), b.getIndexType());
      auto shapeTensor = tensor::FromElementsOp::create(
          b, loc, targetShapeTensorTy, shapeValues);
      return triton::ReshapeOp::create(b, loc, targetTy, v, shapeTensor)
          .getResult();
    }
    if (isa<IndexType>(targetEltTy) || isa<IndexType>(eltTy)) {
      assert((isa<IntegerType>(targetEltTy) || isa<IntegerType>(eltTy)) &&
             "Only cast between index type and integer type");
      return arith::IndexCastOp::create(b, loc, targetTy, v).getResult();
    } else {
      auto targetIntTy = cast<IntegerType>(targetEltTy);
      auto intTy = cast<IntegerType>(eltTy);
      if (targetIntTy.getWidth() > intTy.getWidth())
        return arith::ExtSIOp::create(b, loc, targetTy, v).getResult();
      else
        return arith::TruncIOp::create(b, loc, targetTy, v).getResult();
    }
  } else {
    assert(!shapedTy && "src type rank should be >= target type rank");
    return indexTypeCast(v, targetTy, loc, b);
  }
}

OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
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
        arith::ConstantOp::create(b, loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  return arith::AddIOp::create(b, loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
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
        arith::ConstantOp::create(b, loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = arith::SubIOp::create(b, loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsValue) {
    if (auto lhsOp = lhsValue.getDefiningOp<arith::ConstantOp>()) {
      lhsIntAttr = cast<IntegerAttr>(lhsOp.getValue()).getInt();
    }
  }
  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsValue) {
    if (auto rhsOp = rhsValue.getDefiningOp<arith::ConstantOp>()) {
      rhsIntAttr = cast<IntegerAttr>(rhsOp.getValue()).getInt();
    }
  }

  // shortcut for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }

  if (rhsIntAttr) {
    if (rhsIntAttr.value() == 0)
      return rhs;
    if (rhsIntAttr.value() == 1)
      return lhs;
  }

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() * rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  if (lhsIntAttr) {
    auto lhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  if (rhsIntAttr) {
    auto rhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  return arith::MulIOp::create(b, loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = arith::MinSIOp::create(b, loc, lhsValue, rhsValue);
  return minOp.getResult();
}

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  ensureCmpListener(b);
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        arith::ConstantOp::create(b, loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto maxOp = arith::MaxSIOp::create(b, loc, lhsValue, rhsValue);
  return maxOp.getResult();
}

static bool isShapedValue(OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr)) {
    return isa<ShapedType>(v.getType());
  }
  return false;
}

OpFoldResult compareOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                         const arith::CmpIPredicate pred,
                         const OpFoldResult trueOFR,
                         const OpFoldResult falseOFR, const Location loc,
                         OpBuilder &b) {
  ensureCmpListener(b);
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

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

  if (isShapedValue(lhs) || isShapedValue(rhs) || isShapedValue(trueOFR) ||
      isShapedValue(falseOFR)) {
    llvm::errs()
        << "[compareOFRs] shaped OFR detected, skip creating cmpi/select\n";
    if (auto v = dyn_cast<Value>(lhs)) {
      llvm::errs() << "  lhs type: ";
      v.getType().dump();
      llvm::errs() << "\n";
    }
    if (auto v = dyn_cast<Value>(rhs)) {
      llvm::errs() << "  rhs type: ";
      v.getType().dump();
      llvm::errs() << "\n";
    }
    if (auto v = dyn_cast<Value>(trueOFR)) {
      llvm::errs() << "  trueOFR type: ";
      v.getType().dump();
      llvm::errs() << "\n";
    }
    if (auto v = dyn_cast<Value>(falseOFR)) {
      llvm::errs() << "  falseOFR type: ";
      v.getType().dump();
      llvm::errs() << "\n";
    }

    return falseOFR;
  }

  auto lhsValue = ofrToIndexValue(lhs, loc, b);
  auto rhsValue = ofrToIndexValue(rhs, loc, b);
  auto trueValue = ofrToIndexValue(trueOFR, loc, b);
  auto falseValue = ofrToIndexValue(falseOFR, loc, b);

  auto cmpOp = arith::CmpIOp::create(b, loc, pred, lhsValue, rhsValue);
  auto selectOp = arith::SelectOp::create(b, loc, cmpOp, trueValue, falseValue);
  return selectOp.getResult();
}

// ===----------------------------------------------------------------------===//
// Deep OFR / Value analysis (peers through index_cast chains)
// ===----------------------------------------------------------------------===//

std::optional<int64_t> getConstantIntLike(Value v) {
  if (!v)
    return std::nullopt;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return getConstantIntLike(cast.getIn());
  return std::nullopt;
}

bool isConstZeroOFR(OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 0;
    return false;
  }
  if (auto val = llvm::dyn_cast<Value>(ofr)) {
    auto c = getConstantIntLike(val);
    return c.has_value() && *c == 0;
  }
  return false;
}

bool isConstOneOFR(OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 1;
    return false;
  }
  if (auto val = llvm::dyn_cast<Value>(ofr)) {
    auto c = getConstantIntLike(val);
    return c.has_value() && *c == 1;
  }
  return false;
}

bool sameValueOrEquivalentCast(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (auto cast = lhs.getDefiningOp<arith::IndexCastOp>())
    if (cast.getIn() == rhs)
      return true;
  if (auto cast = rhs.getDefiningOp<arith::IndexCastOp>())
    if (cast.getIn() == lhs)
      return true;
  return false;
}

// --- Internal helpers for sameOFR ---

static Value peelTrivialIndexExpr(Value v) {
  while (v) {
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
      v = cast.getIn();
      continue;
    }
    break;
  }
  return v;
}

static bool sameBoundedSizeExpr(Value lhs, Value rhs);

static bool sameMinSIOp(arith::MinSIOp lhsMin, arith::MinSIOp rhsMin) {
  Value lhsA = peelTrivialIndexExpr(lhsMin.getLhs());
  Value lhsB = peelTrivialIndexExpr(lhsMin.getRhs());
  Value rhsA = peelTrivialIndexExpr(rhsMin.getLhs());
  Value rhsB = peelTrivialIndexExpr(rhsMin.getRhs());
  return (sameBoundedSizeExpr(lhsA, rhsA) && sameBoundedSizeExpr(lhsB, rhsB)) ||
         (sameBoundedSizeExpr(lhsA, rhsB) && sameBoundedSizeExpr(lhsB, rhsA));
}

static bool sameSubIOp(arith::SubIOp lhsSub, arith::SubIOp rhsSub) {
  return sameBoundedSizeExpr(lhsSub.getLhs(), rhsSub.getLhs()) &&
         sameBoundedSizeExpr(lhsSub.getRhs(), rhsSub.getRhs());
}

static bool sameBoundedSizeExpr(Value lhs, Value rhs) {
  lhs = peelTrivialIndexExpr(lhs);
  rhs = peelTrivialIndexExpr(rhs);
  if (!lhs || !rhs)
    return false;
  if (sameValueOrEquivalentCast(lhs, rhs))
    return true;
  if (auto lhsCst = getConstantIntLike(lhs))
    if (auto rhsCst = getConstantIntLike(rhs))
      return *lhsCst == *rhsCst;
  if (auto lhsMin = lhs.getDefiningOp<arith::MinSIOp>())
    if (auto rhsMin = rhs.getDefiningOp<arith::MinSIOp>())
      return sameMinSIOp(lhsMin, rhsMin);
  if (auto lhsSub = lhs.getDefiningOp<arith::SubIOp>())
    if (auto rhsSub = rhs.getDefiningOp<arith::SubIOp>())
      return sameSubIOp(lhsSub, rhsSub);
  return false;
}

bool sameOFR(OpFoldResult lhs, OpFoldResult rhs) {
  auto lhsAttr =
      llvm::dyn_cast_if_present<Attribute>(lhs.dyn_cast<Attribute>());
  auto rhsAttr =
      llvm::dyn_cast_if_present<Attribute>(rhs.dyn_cast<Attribute>());
  if (lhsAttr && rhsAttr)
    return lhsAttr == rhsAttr;

  Value lhsVal = lhs.dyn_cast<Value>();
  Value rhsVal = rhs.dyn_cast<Value>();
  if (lhsVal && rhsVal)
    return sameBoundedSizeExpr(lhsVal, rhsVal);

  if (lhsAttr && rhsVal) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(lhsAttr)) {
      auto c = getConstantIntLike(rhsVal);
      return c.has_value() && *c == intAttr.getInt();
    }
  }
  if (lhsVal && rhsAttr)
    return sameOFR(rhs, lhs);
  return false;
}

} // namespace mlir
