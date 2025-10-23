#include "flagtree/Common/UnifiedHardware.h"

class AscendUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
};

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<AscendUnifiedHardware>();
}
