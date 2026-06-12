#include "RPUExecutableDirectLowering.h"

#include "RPU/IR/Dialect.h"
#include "RPUTransforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rpu {
namespace {

class LowerSupportedTTIRToRPUExecutablePass
    : public PassWrapper<LowerSupportedTTIRToRPUExecutablePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerSupportedTTIRToRPUExecutablePass)

  StringRef getArgument() const final {
    return "rpu-lower-supported-ttir-to-executable";
  }
  StringRef getDescription() const final {
    return "lower supported TTIR directly to executable RPU dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect, triton::TritonDialect>();
  }

  void runOnOperation() final {
    if (failed(lowerSupportedTTIRToRPUExecutable(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerSupportedTTIRToRPUExecutablePass() {
  return std::make_unique<LowerSupportedTTIRToRPUExecutablePass>();
}

} // namespace rpu
} // namespace mlir
