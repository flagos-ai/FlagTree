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
