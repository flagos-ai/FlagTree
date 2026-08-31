#include "flagtree/Common/UnifiedHardware.h"

class TsingmicroUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
  int getDMATag() const override;
  int getSharedMemoryTag() const override;
};

int TsingmicroUnifiedHardware::getDMATag() const { return 11; }
int TsingmicroUnifiedHardware::getSharedMemoryTag() const { return 8; }

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<TsingmicroUnifiedHardware>();
}
