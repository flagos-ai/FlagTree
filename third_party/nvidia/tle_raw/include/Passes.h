#ifndef NV_TLE_RAW_PASSES_H
#define NV_TLE_RAW_PASSES_H

#include "mlir/Pass/Pass.h"
#include "nvidia/tle_raw/include/CompiledRaw.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

#define GEN_PASS_DECL
#include "nvidia/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nvidia/tle_raw/include/Passes.h.inc"

std::unique_ptr<Pass> createNvidiaMaterializeDeferredRaw(
    llvm::StringMap<mlir::triton::nvidia::tle_raw::CompiledRaw> sources);

} // namespace mlir

#endif
