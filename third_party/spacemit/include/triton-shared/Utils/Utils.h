#ifndef TRITON_SHARED_UTILITY_H
#define TRITON_SHARED_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);
Value ensureIndexType(Location loc, Value value, PatternRewriter &rewriter);
Value ofrToIndexValue(const Location loc, const OpFoldResult ofr,
                      PatternRewriter &rewriter);
Value createCeilDivUI(PatternRewriter &rewriter, Location loc, Value dividend,
                      Value divisor);
bool isZeroOFR(mlir::OpFoldResult ofr);
bool areAllZeroOFRs(mlir::ArrayRef<mlir::OpFoldResult> ofrs);
} // namespace triton

} // namespace mlir

#endif // TRITON_SHARED_UTILITY_H
