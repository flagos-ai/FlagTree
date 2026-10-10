#ifndef HCU_TLE_RAW_PASSES_H
#define HCU_TLE_RAW_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "hcu/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hcu/tle_raw/include/Passes.h.inc"

} // namespace mlir

#endif
