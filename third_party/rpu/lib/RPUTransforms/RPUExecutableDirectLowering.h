#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewritePatternSet;

namespace rpu {

LogicalResult lowerSupportedTTIRToRPUExecutable(ModuleOp module);
void populateSupportedTTIRToRPUExecutablePatterns(RewritePatternSet &patterns);

} // namespace rpu
} // namespace mlir
