#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
