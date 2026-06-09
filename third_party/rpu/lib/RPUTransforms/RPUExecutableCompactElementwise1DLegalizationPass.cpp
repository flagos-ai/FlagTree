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

class LegalizeRPUExecutableCompactElementwise1DPass
    : public PassWrapper<LegalizeRPUExecutableCompactElementwise1DPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LegalizeRPUExecutableCompactElementwise1DPass)

  StringRef getArgument() const final {
    return "rpu-legalize-executable-compact-elementwise1d";
  }

  StringRef getDescription() const final {
    return "legalize executable RPU compact Elementwise1D ops to primitive "
           "executable ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<exec::CompactElementwise1DOp> elementwiseOps;
    module.walk(
        [&](exec::CompactElementwise1DOp op) { elementwiseOps.push_back(op); });

    for (exec::CompactElementwise1DOp op : elementwiseOps) {
      StringRef consumer =
          "compact Elementwise1D legalization requires valid high-level op";
      if (failed(exec::expandExecutableCompactElementwise1DOp(builder, op,
                                                              consumer))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(verify(module))) {
      module.emitError(
          "failed to verify executable RPU module after compact Elementwise1D "
          "legalization");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLegalizeRPUExecutableCompactElementwise1DPass() {
  return std::make_unique<LegalizeRPUExecutableCompactElementwise1DPass>();
}

} // namespace rpu
} // namespace mlir
