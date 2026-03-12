#ifndef TRITON_STRUCTURED_TO_MK_CONVERSION_PASSES_H
#define TRITON_STRUCTURED_TO_MK_CONVERSION_PASSES_H

#include "triton-shared/Conversion/StructuredToMK/StructuredToMK.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/StructuredToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
