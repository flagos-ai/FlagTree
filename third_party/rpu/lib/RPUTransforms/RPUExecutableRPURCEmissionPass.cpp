#include "RPUTransforms/Passes.h"

#include "RPU/IR/Dialect.h"
#include "RPUExecutableEmitter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rpu {
namespace {

class EmitRPUExecutableToRPURCPass
    : public PassWrapper<EmitRPUExecutableToRPURCPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitRPUExecutableToRPURCPass)

  StringRef getArgument() const final { return "rpu-emit-executable-to-rpurc"; }

  StringRef getDescription() const final {
    return "emit RPUC .rc source metadata from executable RPU dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::RPUDialect>();
  }

  void runOnOperation() final {
    FailureOr<RPUExecutableEmissionResult> result =
        emitRPUDSLFromExecutableModule(getOperation());
    if (failed(result)) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    module->setAttr("rpu.rpurc.kernel_name",
                    StringAttr::get(context, result->kernelName));
    module->setAttr("rpu.rpurc.source_kind",
                    StringAttr::get(context, result->sourceKind));
    module->setAttr("rpu.rpurc.source",
                    StringAttr::get(context, result->source));
  }
};

} // namespace

std::unique_ptr<Pass> createEmitRPUExecutableToRPURCPass() {
  return std::make_unique<EmitRPUExecutableToRPURCPass>();
}

} // namespace rpu
} // namespace mlir
