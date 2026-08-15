#ifndef TRITON_CONVERSION_TRITONARITHTOLINALG_CONVERSIONTOOLS_H
#define TRITON_CONVERSION_TRITONARITHTOLINALG_CONVERSIONTOOLS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace triton {

static inline SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static inline SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                                    RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1);
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
  return broadcastDims;
}

// Broadcasts input tensor based on TosaToLinalg's broadcastToShape
static inline AffineMap
getBroadcastAffineMap(MLIRContext *context, ArrayRef<int64_t> inputShape,
                      ArrayRef<int64_t> broadcastToShape) {

  assert(broadcastToShape.size() >= inputShape.size());

  // Create affine map and shapes for tensor initialization.
  SmallVector<AffineExpr> outExpr;

  size_t diff = broadcastToShape.size() - inputShape.size();
  for (size_t i = 0; i < broadcastToShape.size(); i++) {
    if (i < diff) {
      continue;
    }
    size_t j = i - diff;
    if (inputShape[j] == 1) {
      // Broadcast singleton dimension
      outExpr.push_back(mlir::getAffineConstantExpr(0, context));
      continue;
    }
    // Non-broadcast case
    outExpr.push_back(mlir::getAffineDimExpr(i, context));
  }
  return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
}

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONARITHTOLINALG_CONVERSIONTOOLS_H
