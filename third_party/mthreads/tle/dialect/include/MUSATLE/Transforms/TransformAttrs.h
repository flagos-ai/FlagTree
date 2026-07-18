#ifndef MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H
#define MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H

#include "llvm/ADT/StringRef.h"

#ifdef __TLE__
namespace mlir::triton::musa_tle {

inline constexpr llvm::StringLiteral
    kMUSATLELocalPointerAsyncStoreAttr("musa_tle.local_ptr_async_store");

} // namespace mlir::triton::musa_tle
#endif // __TLE__

#endif // MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H
