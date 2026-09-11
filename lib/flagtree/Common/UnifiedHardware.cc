#include "flagtree/Common/UnifiedHardware.h"
#include <memory>
namespace mlir {
namespace flagtree {

bool UnifiedHardware::isRegistered() const {
#ifdef FLAGTREE_BACKEND
  return true;
#else
  return false;
#endif
}

int UnifiedHardware::getDMATag() const { return 0; }

int UnifiedHardware::getSharedMemoryTag() const { return 0; }

bool UnifiedHardware::getIncubatedTag() const { return false; }

std::string UnifiedHardware::getReduceStrategy() const {
  return "linalg_reduce";
}

std::string UnifiedHardware::getFlagTreeBackend() const { return "default"; }

__attribute__((weak)) std::unique_ptr<UnifiedHardware>
createUnifiedHardwareManager() {
  return std::make_unique<UnifiedHardware>();
}

} // namespace flagtree
} // namespace mlir
