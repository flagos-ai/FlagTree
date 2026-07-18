#include "test/include/Analysis/TestAxisInfo.h"
#include "third_party/hcu/include/Analysis/AxisInfoExt.h"

namespace {

struct HCUTestAxisInfoPass : public mlir::test::TestAxisInfoPass {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCUTestAxisInfoPass);

  StringRef getArgument() const final { return "test-print-hcu-alignment"; }

protected:
  ModuleAxisInfoAnalysis getAnalysis(ModuleOp moduleOp) const final {
    return HCU::ModuleAxisInfoAnalysis(moduleOp);
  }
};
} // namespace

namespace mlir::test {
void registerHCUTestAlignmentPass() { PassRegistration<HCUTestAxisInfoPass>(); }
} // namespace mlir::test
