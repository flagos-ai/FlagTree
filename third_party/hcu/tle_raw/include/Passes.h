#ifndef HCU_TLE_RAW_PASSES_H
#define HCU_TLE_RAW_PASSES_H

#include "hcu/tle_raw/include/CompiledRaw.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

#define GEN_PASS_DECL
#include "hcu/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hcu/tle_raw/include/Passes.h.inc"

std::unique_ptr<Pass> createHCUMaterializeDeferredRaw(
    llvm::StringMap<mlir::triton::HCU::tle_raw::CompiledRaw> sources);

} // namespace mlir

#endif
