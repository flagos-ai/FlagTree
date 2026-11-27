#ifndef TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_STRUCTURED_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
