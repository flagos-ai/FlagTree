#ifndef TRITON_ADD_TARGET_DESCRIPTION_CONVERSION_PASSES_H
#define TRITON_ADD_TARGET_DESCRIPTION_CONVERSION_PASSES_H

#include "triton-shared/Conversion/AddTargetDescription/AddTargetDescription.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/AddTargetDescription/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
