#ifndef NV_TLE_RAW_PASSES_H
#define NV_TLE_RAW_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "nvidia/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nvidia/tle_raw/include/Passes.h.inc"

} // namespace mlir

#endif
