#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
