// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
