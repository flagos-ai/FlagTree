#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES_H
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
