#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
namespace rpu {

namespace plan {
class KernelOp;
} // namespace plan

struct RPUDSLEmissionResult {
  std::string kernelName;
  std::string pattern;
  std::string source;
};

struct RPUPlanKernelSummary {
  std::string kernelName;
  std::string pattern;
};

std::vector<std::string> directRPUDSLSupportedPatterns();
bool isDirectRPUDSLSupportedPattern(llvm::StringRef pattern);
FailureOr<RPUPlanKernelSummary>
getRPUPlanKernelSummaryFromKernelOp(plan::KernelOp op);
FailureOr<RPUPlanKernelSummary>
getRPUPlanKernelSummaryFromModule(ModuleOp module);
FailureOr<RPUDSLEmissionResult> emitRPUDSLFromKernelOp(plan::KernelOp op);
FailureOr<RPUDSLEmissionResult> emitRPUDSLFromModule(ModuleOp module);

} // namespace rpu
} // namespace mlir
