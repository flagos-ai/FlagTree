#ifndef MTHREADS_MUSATLE_FRONTEND_PASSES_H
#define MTHREADS_MUSATLE_FRONTEND_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "MUSATLE/Frontend/Passes.h.inc"

} // namespace mlir

namespace mlir {
// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "MUSATLE/Frontend/Passes.h.inc"
} // namespace mlir

#endif // MTHREADS_MUSATLE_FRONTEND_PASSES_H
