#pragma once

#include "RPUExecutableEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace mlir {
namespace rpu {

bool supportsRPUPlanExecutableLowering(ModuleOp module);
FailureOr<std::string>
describeRPUPlanExecutableLoweringFailure(ModuleOp module);
FailureOr<OwningOpRef<ModuleOp>>
lowerRPUPlanToExecutableModuleOp(ModuleOp module);
FailureOr<std::string> lowerRPUPlanToExecutableModule(ModuleOp module);
FailureOr<std::string> lowerRPUPlanAddToExecutableModule(ModuleOp module);

} // namespace rpu
} // namespace mlir
