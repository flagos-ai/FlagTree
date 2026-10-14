#include "hcu/tle_raw/include/DeferredRawSourceRegistry.h"

namespace mlir::triton::HCU::tle_raw {

static llvm::StringMap<DeferredRawSourceEntry> deferredRawSourceRegistry;

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry() {
  return deferredRawSourceRegistry;
}

void clearDeferredRawSourceRegistry() { deferredRawSourceRegistry.clear(); }

} // namespace mlir::triton::HCU::tle_raw
