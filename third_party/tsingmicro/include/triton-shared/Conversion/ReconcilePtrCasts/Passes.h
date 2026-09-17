#ifndef RECONCILE_PTR_CASTS_CONVERSION_PASSES_H
#define RECONCILE_PTR_CASTS_CONVERSION_PASSES_H

#include "triton-shared/Conversion/ReconcilePtrCasts/ReconcilePtrCasts.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/ReconcilePtrCasts/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
