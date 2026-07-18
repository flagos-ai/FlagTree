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

#include "RPU/IR/Dialect.h"
#include "RPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace rpu {
namespace {

struct ExecutableHighLevelLegalizationEntry {
  const char *opName;
  std::unique_ptr<Pass> (*createPass)();
};

static const ExecutableHighLevelLegalizationEntry
    kExecutableHighLevelLegalizationEntries[] = {
        {"rpu.compact_elementwise1d",
         createLegalizeRPUExecutableCompactElementwise1DPass},
        {"rpu.dot", createLegalizeRPUExecutableDotPass},
        {"rpu.softmax", createLegalizeRPUExecutableSoftmaxPass},
        {"rpu.elementwise16_value_map",
         createLegalizeRPUExecutableValueMapsPass},
};

static bool hasHighLevelLegalizationEntry(llvm::StringRef opName) {
  for (const ExecutableHighLevelLegalizationEntry &entry :
       kExecutableHighLevelLegalizationEntries) {
    if (opName == entry.opName)
      return true;
  }
  return false;
}

static bool highLevelLegalizationRegistryMatchesIRContract() {
  for (llvm::StringLiteral opName :
       exec::getHighLevelLegalizableExecutableOpNames()) {
    if (!hasHighLevelLegalizationEntry(opName))
      return false;
  }
  for (const ExecutableHighLevelLegalizationEntry &entry :
       kExecutableHighLevelLegalizationEntries) {
    if (!exec::isHighLevelLegalizableExecutableOpName(entry.opName))
      return false;
  }
  return true;
}

} // namespace

void addRPUExecutableHighLevelLegalizationPipeline(OpPassManager &pm) {
  if (!highLevelLegalizationRegistryMatchesIRContract())
    llvm::report_fatal_error(
        "RPU executable high-level legalization registry must match IR op "
        "lowering class contract");
  for (const ExecutableHighLevelLegalizationEntry &entry :
       kExecutableHighLevelLegalizationEntries)
    pm.addPass(entry.createPass());
  pm.addPass(createVerifyRPUExecutableRenderablePass());
}

} // namespace rpu
} // namespace mlir
