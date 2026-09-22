#include "iluvatar/tle_raw/include/DeferredRawSourceRegistry.h"

namespace mlir::triton::iluvatar::tle_raw {

static llvm::StringMap<DeferredRawSourceEntry> gDeferredRawSourceRegistry;

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry() {
  return gDeferredRawSourceRegistry;
}

void clearDeferredRawSourceRegistry() { gDeferredRawSourceRegistry.clear(); }

} // namespace mlir::triton::iluvatar::tle_raw
