//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALG_TILING_PASSES_H
#define TRITON_CONVERSION_LINALG_TILING_PASSES_H

#include "tsingmicro-tx81/Conversion/LinalgTiling/LinalgTiling.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/LinalgTiling/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALG_TILING_PASSES_H
