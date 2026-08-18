#ifndef TRITON_XSMT_TO_LINALG_CONVERSION_PASSES_H
#define TRITON_XSMT_TO_LINALG_CONVERSION_PASSES_H

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
