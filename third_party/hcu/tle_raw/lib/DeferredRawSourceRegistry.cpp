#include "hcu/tle_raw/include/DeferredRawSourceRegistry.h"

namespace mlir::triton::hcu::tle_raw {

static llvm::StringMap<DeferredRawSourceEntry> gDeferredRawSourceRegistry;

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry() {
  return gDeferredRawSourceRegistry;
}

void clearDeferredRawSourceRegistry() { gDeferredRawSourceRegistry.clear(); }

} // namespace mlir::triton::hcu::tle_raw
