#include "RPU/IR/Dialect.h"
#include "RPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir {
namespace rpu {
namespace {

class LegalizeRPUExecutableDotPass
    : public PassWrapper<LegalizeRPUExecutableDotPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeRPUExecutableDotPass)

  StringRef getArgument() const final { return "rpu-legalize-executable-dot"; }

  StringRef getDescription() const final {
    return "legalize executable RPU dot ops to primitive executable ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<exec::DotOp> dotOps;
    module.walk([&](exec::DotOp op) { dotOps.push_back(op); });

    for (exec::DotOp op : dotOps) {
      StringRef consumer = "dot legalization requires valid high-level op";
      if (failed(exec::expandExecutableDotOp(builder, op, consumer))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(verify(module))) {
      module.emitError(
          "failed to verify executable RPU module after dot legalization");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLegalizeRPUExecutableDotPass() {
  return std::make_unique<LegalizeRPUExecutableDotPass>();
}

} // namespace rpu
} // namespace mlir
