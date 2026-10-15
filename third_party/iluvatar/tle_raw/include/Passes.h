#ifndef ILUVATAR_TLE_RAW_PASSES_H
#define ILUVATAR_TLE_RAW_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "iluvatar/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "iluvatar/tle_raw/include/Passes.h.inc"

} // namespace mlir

#endif
