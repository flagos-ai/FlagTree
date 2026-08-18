#ifndef TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES_H
#define TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
