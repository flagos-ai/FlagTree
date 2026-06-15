#include "RPUExecutableLowering.h"

#include "RPU/IR/Dialect.h"
#include "RPUExecutableEmitter.h"
#include "RPUPlan/IR/Dialect.h"
#include "RPUPlanExecutableBridge.h"
#include "RPUTransforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rpu {

namespace {

FailureOr<plan::KernelOp> getSinglePlanKernel(ModuleOp module) {
  llvm::SmallVector<plan::KernelOp> plans;
  module.walk([&](plan::KernelOp op) { plans.push_back(op); });
  if (plans.empty()) {
    module.emitError(
        "RPU executable lowering requires one rpu_plan.kernel, found none");
    return failure();
  }
  if (plans.size() > 1) {
    module.emitError(
        "RPU executable lowering requires one rpu_plan.kernel, found multiple");
    return failure();
  }
  return plans.front();
}

class ConvertRPUPlanToRPUExecutablePass
    : public PassWrapper<ConvertRPUPlanToRPUExecutablePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertRPUPlanToRPUExecutablePass)

  StringRef getArgument() const final {
    return "rpu-convert-plan-to-executable";
  }
  StringRef getDescription() const final {
    return "convert supported rpu_plan.kernel ops to executable RPU dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plan::RPUPlanDialect, exec::RPUDialect,
                    triton::TritonDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(getSinglePlanKernel(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    populateRPUPlanToExecutablePatterns(patterns);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<plan::KernelOp> remainingPlans;
    module.walk([&](plan::KernelOp op) { remainingPlans.push_back(op); });
    if (remainingPlans.empty())
      return;
    if (remainingPlans.size() > 1) {
      module.emitError("RPU executable lowering requires one rpu_plan.kernel, "
                       "found multiple");
      signalPassFailure();
      return;
    }

    plan::KernelOp remaining = remainingPlans.front();
    remaining.emitError(
        "RPU executable builder does not support this rpu_plan.kernel: ")
        << describeRPUPlanKernelExecutableBuildFailure(remaining);
    signalPassFailure();
  }
};

static FailureOr<OwningOpRef<ModuleOp>>
lowerRPUPlanModuleToExecutableWithPass(ModuleOp module) {
  Operation *clone = module->clone();
  OwningOpRef<ModuleOp> executableModule(cast<ModuleOp>(clone));
  PassManager pm(module.getContext());
  pm.addPass(createConvertRPUPlanToRPUExecutablePass());
  if (failed(pm.run(*executableModule))) {
    module.emitError("RPU executable lowering pass failed");
    return failure();
  }
  if (failed(verify(*executableModule))) {
    executableModule->emitError(
        "RPU executable lowering produced unverified executable module");
    return failure();
  }
  return executableModule;
}

} // namespace

std::unique_ptr<Pass> createConvertRPUPlanToRPUExecutablePass() {
  return std::make_unique<ConvertRPUPlanToRPUExecutablePass>();
}

bool supportsRPUPlanExecutableLowering(ModuleOp module) {
  FailureOr<plan::KernelOp> op = getSinglePlanKernel(module);
  if (failed(op))
    return false;
  return supportsRPUPlanKernelExecutableBuild(*op);
}

FailureOr<std::string>
describeRPUPlanExecutableLoweringFailure(ModuleOp module) {
  FailureOr<plan::KernelOp> op = getSinglePlanKernel(module);
  if (failed(op))
    return failure();
  if (supportsRPUPlanKernelExecutableBuild(*op))
    return std::string();
  return describeRPUPlanKernelExecutableBuildFailure(*op);
}

FailureOr<OwningOpRef<ModuleOp>>
lowerRPUPlanToExecutableModuleOp(ModuleOp module) {
  FailureOr<plan::KernelOp> op = getSinglePlanKernel(module);
  if (failed(op))
    return failure();

  return lowerRPUPlanModuleToExecutableWithPass(module);
}

FailureOr<std::string> lowerRPUPlanToExecutableModule(ModuleOp module) {
  FailureOr<OwningOpRef<ModuleOp>> executableModule =
      lowerRPUPlanToExecutableModuleOp(module);
  if (failed(executableModule))
    return failure();

  std::string result;
  llvm::raw_string_ostream os(result);
  (*executableModule)->print(os);
  os << "\n";
  return os.str();
}

FailureOr<std::string> lowerRPUPlanAddToExecutableModule(ModuleOp module) {
  FailureOr<plan::KernelOp> op = getSinglePlanKernel(module);
  if (failed(op))
    return failure();

  if (!supportsRPUPlanKernelExecutableBuildKind(*op, "add")) {
    (*op).emitError(
        "RPU executable add lowering does not support this rpu_plan.kernel: ")
        << describeRPUPlanKernelExecutableBuildFailure(*op);
    return failure();
  }
  FailureOr<OwningOpRef<ModuleOp>> executableModule =
      lowerRPUPlanModuleToExecutableWithPass(module);
  if (failed(executableModule))
    return failure();

  std::string result;
  llvm::raw_string_ostream os(result);
  (*executableModule)->print(os);
  os << "\n";
  return os.str();
}

} // namespace rpu
} // namespace mlir
