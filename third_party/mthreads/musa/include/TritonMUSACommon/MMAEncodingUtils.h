#ifndef TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H
#define TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H

#include "TritonMUSACommon/MusaArchTraits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::musa {
namespace ttg = mlir::triton::gpu;

inline bool supportsMusaWmmaEncoding(ttg::MUSAWmmaEncodingAttr encoding) {
  return encoding && getMusaArchFromWmmaVersion(encoding.getVersionMajor(),
                                                encoding.getVersionMinor())
                         .has_value();
}

inline bool supportsMusaSqmmaEncoding(ttg::MUSASqmmaEncodingAttr encoding) {
  if (!encoding)
    return false;
  auto arch = getMusaArchFromSqmmaVersion(encoding.getVersionMajor(),
                                          encoding.getVersionMinor());
  return arch && getMusaSqmmaArchTraits(*arch) != nullptr;
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H
