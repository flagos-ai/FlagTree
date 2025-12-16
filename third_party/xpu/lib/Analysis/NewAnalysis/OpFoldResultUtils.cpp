#include "triton/Analysis/NewAnalysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

// std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
//   if (ofr.is<Attribute>() && isa<IntegerAttr>(ofr.get<Attribute>()))
//     return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
//
//   return std::nullopt;
// }
//
// bool hasConstZero(const OpFoldResult ofr) {
//   auto intAttr = getIntAttr(ofr);
//   if (intAttr.has_value()) {
//     if (intAttr.value() == 0) {
//       return true;
//     }
//     return false;
//   }
//
//   auto val = dyn_cast<Value>(ofr);
//   assert(val);
//   auto constOp = val.getDefiningOp<arith::ConstantOp>();
//   if (!constOp)
//     return false;
//
//   intAttr = getIntAttr(constOp.getValue());
//   if (intAttr.has_value()) {
//     if (intAttr.value() == 0) {
//       return true;
//     }
//     return false;
//   }
//
//   return false;
// }
//
// bool hasConstOne(const OpFoldResult ofr) {
//   auto intAttr = getIntAttr(ofr);
//   if (intAttr.has_value()) {
//     if (intAttr.value() == 1) {
//       return true;
//     }
//     return false;
//   }
//
//   auto val = dyn_cast<Value>(ofr);
//   assert(val);
//   auto constOp = val.getDefiningOp<arith::ConstantOp>();
//   if (!constOp)
//     return false;
//
//   intAttr = getIntAttr(constOp.getValue());
//   if (intAttr.has_value()) {
//     if (intAttr.value() == 1) {
//       return true;
//     }
//     return false;
//   }
//
//   return false;
// }
//
// Value ofrToIndexValue(const OpFoldResult ofr, const Location loc,
//                       OpBuilder &b) {
//   if (Value val = dyn_cast<Value>(ofr)) {
//     assert(val.getType().isIndex() && "Provided ofr is of type index");
//     return val;
//   }
//
//   auto intVal = getIntAttr(ofr);
//   if (intVal.has_value()) {
//     return b.create<arith::ConstantOp>(loc, b.getIndexAttr(intVal.value()));
//   }
//   llvm_unreachable("Unexpected OpFoldResult state");
//   return nullptr;
// }
//
// SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
//                                      const Location loc, OpBuilder &b) {
//   return llvm::to_vector<4>(
//       llvm::map_range(ofrs, [&](OpFoldResult ofr) -> Value {
//         return ofrToIndexValue(ofr, loc, b);
//       }));
// }
//
// OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
//                      const Location loc, OpBuilder &b) {
//   auto lhsIntAttr = getIntAttr(lhs);
//   auto rhsIntAttr = getIntAttr(rhs);
//
//   // shortcut for special cases
//   if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
//     return lhs;
//   if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
//     return rhs;
//
//   // both lhs and rhs are constants, return result directly
//   if (lhsIntAttr && rhsIntAttr)
//     return b.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());
//
//   // otherwise, need to create instructions to calculate new attribute value
//   auto lhsValue = dyn_cast<Value>(lhs);
//   if (lhsIntAttr) {
//     auto lhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
//     lhsValue = lhsOp.getResult();
//   } else {
//     assert(isa<IndexType>(lhsValue.getType()));
//   }
//
//   auto rhsValue = dyn_cast<Value>(rhs);
//   if (rhsIntAttr) {
//     auto rhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
//     rhsValue = rhsOp.getResult();
//   } else {
//     assert(isa<IndexType>(lhsValue.getType()));
//   }
//
//   return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
// }
//
// OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
//                      const Location loc, OpBuilder &b) {
//   auto lhsIntAttr = getIntAttr(lhs);
//   auto rhsIntAttr = getIntAttr(rhs);
//
//   // shortcut for special cases
//   if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
//     return lhs;
//
//   // both lhs and rhs are constants, return result directly
//   if (lhsIntAttr && rhsIntAttr)
//     return b.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());
//
//   // otherwise, need to create instructions to calculate new attribute value
//   auto lhsValue = dyn_cast<Value>(lhs);
//   if (lhsIntAttr) {
//     auto lhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
//     lhsValue = lhsOp.getResult();
//   }
//
//   auto rhsValue = dyn_cast<Value>(rhs);
//   if (rhsIntAttr) {
//     auto rhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
//     rhsValue = rhsOp.getResult();
//   }
//
//   auto sumOp = b.create<arith::SubIOp>(loc, lhsValue, rhsValue);
//   return sumOp.getResult();
// }
//
// OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
//                          const Location loc, OpBuilder &b) {
//   auto lhsIntAttr = getIntAttr(lhs);
//
//   auto rhsIsConst = false;
//   // if rhs is not a const, use max value since min is used to represent
//   // dynamic size or stride
//   auto rhsConstValue = std::numeric_limits<int64_t>::max();
//   auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
//   if (rhsOp) {
//     rhsIsConst = true;
//     rhsConstValue = cast<IntegerAttr>(rhsOp.getValue()).getInt();
//   }
//
//   // shortcuts for special cases
//   if (lhsIntAttr) {
//     if (lhsIntAttr.value() == 0)
//       return lhs;
//     if (lhsIntAttr.value() == 1)
//       return rhs;
//   }
//   if (rhsIsConst) {
//     if (rhsConstValue == 0)
//       return rhsOp.getResult();
//     if (rhsConstValue == 1)
//       return lhs;
//   }
//
//   // 0. both lhs and rhs are constants
//   if (lhsIntAttr && rhsIsConst)
//     return b.getIndexAttr(lhsIntAttr.value() * rhsConstValue);
//
//   // 1. if lhs is constant but rhs is not
//   if (lhsIntAttr && !rhsIsConst) {
//     auto lhsConstOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
//     auto mulOp = b.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
//     return mulOp.getResult();
//   }
//
//   // 2. if lhs is not constant
//   assert(!lhsIntAttr);
//   auto mulOp = b.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs);
//   return mulOp.getResult();
// }
//
// OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
//                      const Location loc, OpBuilder &b) {
//   auto lhsIntAttr = getIntAttr(lhs);
//   auto rhsIntAttr = getIntAttr(rhs);
//
//   // both lhs and rhs are constants, return result directly
//   if (lhsIntAttr && rhsIntAttr)
//     return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));
//
//   // otherwise, need to create instructions to calculate new attribute value
//   auto lhsValue = dyn_cast<Value>(lhs);
//   if (lhsIntAttr) {
//     auto lhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
//     lhsValue = lhsOp.getResult();
//   }
//
//   auto rhsValue = dyn_cast<Value>(rhs);
//   if (rhsIntAttr) {
//     auto rhsOp =
//         b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
//     rhsValue = rhsOp.getResult();
//   }
//
//   auto minOp = b.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
//   return minOp.getResult();
// }
//
} // namespace mlir
