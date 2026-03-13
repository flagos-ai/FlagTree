#ifndef TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonToMK/TritonToMKPatterns.hpp"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
