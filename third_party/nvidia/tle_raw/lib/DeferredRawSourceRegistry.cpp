#include "nvidia/tle_raw/include/DeferredRawSourceRegistry.h"

namespace mlir::triton::nvidia::tle_raw {

static llvm::StringMap<DeferredRawSourceEntry> gDeferredRawSourceRegistry;

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry() {
  return gDeferredRawSourceRegistry;
}

void clearDeferredRawSourceRegistry() { gDeferredRawSourceRegistry.clear(); }

} // namespace mlir::triton::nvidia::tle_raw
