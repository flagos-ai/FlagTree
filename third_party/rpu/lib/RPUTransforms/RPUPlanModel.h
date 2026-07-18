#pragma once

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace rpu {

namespace plan {
class KernelOp;
} // namespace plan

struct RPUPlan {
  int version = 1;
  std::string kernelName;
  std::vector<llvm::json::Object> signatureParams;
  std::string returnType = "void";
  std::string pattern;
  std::map<std::string, int64_t> shape;
  std::map<std::string, int64_t> args;
  llvm::json::Object layout;
  llvm::json::Object mask;
  std::vector<std::string> requiredDslFeatures;
  llvm::json::Object emission;
};

// RPUPlan is a C++ conversion DTO. Stable JSON export must go through
// rpu_plan.kernel so the dialect op remains the internal source of truth.
std::optional<RPUPlan> rpuPlanFromKernelOp(plan::KernelOp op);
std::optional<std::string> serializeRPUPlanKernelOpToJson(plan::KernelOp op);

} // namespace rpu
} // namespace mlir
