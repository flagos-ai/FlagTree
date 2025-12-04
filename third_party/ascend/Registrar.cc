#include "flagtree/Common/UnifiedHardware.h"

class AscendUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
  bool getIncubatedTag() const override;
};

bool AscendUnifiedHardware::getIncubatedTag() const { return 1; }

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<AscendUnifiedHardware>();
}
