#ifndef ILUVATAR_TLE_RAW_PASSES_H
#define ILUVATAR_TLE_RAW_PASSES_H

#include "iluvatar/tle_raw/include/CompiledRaw.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

#define GEN_PASS_DECL
#include "iluvatar/tle_raw/include/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "iluvatar/tle_raw/include/Passes.h.inc"

std::unique_ptr<Pass> createIluvatarMaterializeDeferredRaw(
    llvm::StringMap<mlir::triton::iluvatar::tle_raw::CompiledRaw> sources);

} // namespace mlir

#endif
