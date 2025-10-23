#ifndef TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H

#include "TritonToLinalgPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "ascend/triton-adapter/include/TritonToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_LINALG_CONVERSION_PASSES_H