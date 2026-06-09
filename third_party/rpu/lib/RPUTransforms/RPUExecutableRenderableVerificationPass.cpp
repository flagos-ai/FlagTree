#include "RPUTransforms/Passes.h"

#include "RPU/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rpu {
namespace {

class VerifyRPUExecutableRenderablePass
    : public PassWrapper<VerifyRPUExecutableRenderablePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyRPUExecutableRenderablePass)

  StringRef getArgument() const final {
    return "rpu-verify-executable-renderable";
  }

  StringRef getDescription() const final {
    return "verify executable RPU dialect is renderable before RPURC emission";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    if (failed(exec::verifyExecutableModuleRenderable(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVerifyRPUExecutableRenderablePass() {
  return std::make_unique<VerifyRPUExecutableRenderablePass>();
}

} // namespace rpu
} // namespace mlir
