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
