#ifndef ILUVATAR_TLE_PASSES_H
#define ILUVATAR_TLE_PASSES_H

#include "IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DECL
#include "Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

} // namespace mlir::triton::iluvatar_tle

#endif // ILUVATAR_TLE_PASSES_H
