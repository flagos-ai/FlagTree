#ifndef TRITON_ADAPTER_TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES_H

#include "BubbleUpOperation.h"
#include "UnstructureConversionPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "ascend/triton-adapter/include/TritonToUnstructure/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES_H