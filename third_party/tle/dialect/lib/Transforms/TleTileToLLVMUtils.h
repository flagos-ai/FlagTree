#ifndef TLE_TILE_TO_LLVM_UTILS_H
#define TLE_TILE_TO_LLVM_UTILS_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace mlir::triton::tle {

template <typename T1, typename T2, typename BinaryOp>
llvm::SmallVector<T2> multiDimElementwise(llvm::ArrayRef<T1> lhs, llvm::ArrayRef<T2> rhs, BinaryOp op) {
    assert(lhs.size() == rhs.size() && "Dimensions must match");
    llvm::SmallVector<T2> result;
    result.reserve(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i)
        result.push_back(static_cast<T2>(op(lhs[i], rhs[i])));
    return result;
}

llvm::SmallVector<unsigned> getCTATileOrder(::mlir::RankedTensorType type);

llvm::SmallVector<unsigned> delinearize(unsigned linearIndex,
                                        llvm::ArrayRef<unsigned> shape,
                                        llvm::ArrayRef<unsigned> order);

unsigned linearize(llvm::ArrayRef<unsigned> coords,
                   llvm::ArrayRef<unsigned> shape,
                   llvm::ArrayRef<unsigned> order);

llvm::SmallVector<unsigned> getShapePerCTATile(::mlir::RankedTensorType type);

llvm::SmallVector<::mlir::Value> computeThreadOffsets(
    ::mlir::Location loc,
    ::mlir::ConversionPatternRewriter &rewriter,
    ::mlir::RankedTensorType tensorType);

} // namespace mlir::triton::tle

#endif
