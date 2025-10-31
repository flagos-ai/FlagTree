#include "flagtree/Common/UnifiedHardware.h"

class AipuUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
  int getDMATag() const override;
  int getSharedMemoryTag() const override;
};

int AipuUnifiedHardware::getDMATag() const { return 11; }
int AipuUnifiedHardware::getSharedMemoryTag() const { return 8; }

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<AipuUnifiedHardware>();
}
