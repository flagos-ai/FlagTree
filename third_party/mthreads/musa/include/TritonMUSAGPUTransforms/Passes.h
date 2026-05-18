#ifndef TRITON_THIRD_PARTY_MUSA_INCLUDE_TRITONMUSAGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_MUSA_INCLUDE_TRITONMUSAGPUTRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonMUSAGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir {
// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMUSAGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_MUSA_INCLUDE_TRITONMUSAGPUTRANSFORMS_PASSES_H_
