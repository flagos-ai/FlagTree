#ifndef TRITONMUSA_COMMON_CONVERT_LAYOUT_UTILS_H
#define TRITONMUSA_COMMON_CONVERT_LAYOUT_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton::musa {

bool isMusaSqmmaLike(Attribute layout);
bool isPlainBlockedLike(Attribute layout);
bool useMusaReplicatedScratch(Attribute srcLayout, Attribute dstLayout);
bool isMusaSqmmaAccumulatorToBlockedLike(Attribute srcLayout,
                                         Attribute dstLayout);
bool useMusaSqmmaBlockSwizzling(RankedTensorType srcTy, RankedTensorType dstTy);
bool useConservativeCarrierScratch(RankedTensorType srcTy,
                                   RankedTensorType dstTy);
bool useMusaGenericBlockSwizzling(RankedTensorType srcTy,
                                  RankedTensorType dstTy);
bool musaConvertLayoutHasLoweringPath(RankedTensorType srcTy,
                                      RankedTensorType dstTy);

} // namespace mlir::triton::musa

#endif
