#ifndef TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES_H
#define TRITON_NVIDIA_COMMON_IR_TO_TTGIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "nvidia/include/CommonIRToTTGIR/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nvidia/include/CommonIRToTTGIR/Passes.h.inc"

} // namespace mlir::triton

#endif
