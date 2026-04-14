#ifndef TLE_RAW_CONVERSION_TLETOLLVM_REMOTEPOINTERUTILS_H
#define TLE_RAW_CONVERSION_TLETOLLVM_REMOTEPOINTERUTILS_H

#include "mlir/IR/Value.h"

namespace mlir::triton::tle {

unsigned inferTlePointerLayoutVectorHint(Value ptr);

} // namespace mlir::triton::tle

#endif
