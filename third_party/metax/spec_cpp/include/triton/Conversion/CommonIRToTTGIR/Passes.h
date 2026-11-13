#ifndef TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES_H
#define TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::metax {

#define GEN_PASS_DECL
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

} // namespace mlir::triton::metax

#endif
