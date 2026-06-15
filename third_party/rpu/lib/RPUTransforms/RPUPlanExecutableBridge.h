#ifndef RPU_PLAN_EXECUTABLE_BRIDGE_H
#define RPU_PLAN_EXECUTABLE_BRIDGE_H

#include "RPUPlan/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {

class RewritePatternSet;

namespace rpu {

bool supportsRPUPlanKernelExecutableBuild(plan::KernelOp op);

bool supportsRPUPlanKernelExecutableBuildKind(plan::KernelOp op,
                                              llvm::StringRef kind);

std::string describeRPUPlanKernelExecutableBuildFailure(plan::KernelOp op);

void populateRPUPlanToExecutablePatterns(RewritePatternSet &patterns);

} // namespace rpu
} // namespace mlir

#endif // RPU_PLAN_EXECUTABLE_BRIDGE_H
