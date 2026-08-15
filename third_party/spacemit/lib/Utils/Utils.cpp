#include "mlir/Dialect/Arith/IR/Arith.h" // arith::ConstantIndexOp
#include "mlir/IR/BuiltinAttributes.h"   // IntegerAttr
#include "mlir/IR/PatternMatch.h"        // matchPattern, m_Zero
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h" // llvm::all_of

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

Value ensureIndexType(Location loc, Value value, PatternRewriter &rewriter) {
  auto indexType = rewriter.getIndexType();
  if (value.getType() == indexType) {
    return value;
  }
  return arith::IndexCastOp::create(rewriter, loc, indexType, value);
}

Value ofrToIndexValue(const Location loc, const OpFoldResult ofr,
                      PatternRewriter &rewriter) {
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                       val);
    }
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

Value createCeilDivUI(PatternRewriter &rewriter, Location loc, Value dividend,
                      Value divisor) {
  auto indexType = rewriter.getIndexType();
  Value dividendIndex = ensureIndexType(loc, dividend, rewriter);
  Value divisorIndex = ensureIndexType(loc, divisor, rewriter);

  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto sum = arith::AddIOp::create(rewriter, loc, dividendIndex, divisorIndex);
  auto numerator = arith::SubIOp::create(rewriter, loc, sum, one);
  return arith::DivUIOp::create(rewriter, loc, numerator, divisorIndex);
}

bool isZeroOFR(mlir::OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<mlir::Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    return false;
  }

  mlir::Value v = llvm::cast<mlir::Value>(ofr);

  if (mlir::matchPattern(v, mlir::m_Zero()))
    return true;

  if (auto cstIdx = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return cstIdx.value() == 0;

  return false;
}

bool areAllZeroOFRs(mlir::ArrayRef<mlir::OpFoldResult> ofrs) {
  return llvm::all_of(ofrs, [](mlir::OpFoldResult x) { return isZeroOFR(x); });
}

} // namespace triton

} // namespace mlir
