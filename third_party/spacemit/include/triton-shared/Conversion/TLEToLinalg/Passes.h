#ifndef TRITON_TLE_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_TLE_TO_LINALG_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TLEToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
