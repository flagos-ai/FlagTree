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

class LegalizeRPUExecutableValueMapsPass
    : public PassWrapper<LegalizeRPUExecutableValueMapsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LegalizeRPUExecutableValueMapsPass)

  StringRef getArgument() const final {
    return "rpu-legalize-executable-value-maps";
  }

  StringRef getDescription() const final {
    return "legalize executable RPU value-map ops to primitive executable ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<exec::Elementwise16ValueMapOp> valueMaps;
    module.walk([&](exec::Elementwise16ValueMapOp valueMap) {
      valueMaps.push_back(valueMap);
    });

    for (exec::Elementwise16ValueMapOp valueMap : valueMaps) {
      StringRef consumer =
          "executable value-map legalization requires valid value-map op";
      if (failed(exec::expandExecutableElementwise16ValueMapOp(
              builder, valueMap, consumer))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(verify(module))) {
      module.emitError("failed to verify executable RPU module after value-map "
                       "legalization");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLegalizeRPUExecutableValueMapsPass() {
  return std::make_unique<LegalizeRPUExecutableValueMapsPass>();
}

} // namespace rpu
} // namespace mlir
