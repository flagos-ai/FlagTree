#ifndef TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H
#define TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::musa {
namespace ttg = mlir::triton::gpu;

inline constexpr unsigned kMusaPH1VersionMajor = 3;

// Keep backend support policy out of the public TTG verifier. The TTG encoding
// carries version metadata, while the MUSA backend decides which generations it
// can lower.
inline bool supportsMusaWmmaEncoding(ttg::MUSAWmmaEncodingAttr encoding) {
  return encoding && encoding.getVersionMajor() == kMusaPH1VersionMajor;
}

inline bool supportsMusaSqmmaEncoding(ttg::MUSASqmmaEncodingAttr encoding) {
  return encoding && encoding.getVersionMajor() == kMusaPH1VersionMajor;
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MMA_ENCODING_UTILS_H
