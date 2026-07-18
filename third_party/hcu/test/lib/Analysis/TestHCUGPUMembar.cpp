#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "hcu/lib/TritonHCUGPUToLLVM/AsyncUtility.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestHCUGPUMembarPass
    : public PassWrapper<TestHCUGPUMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUGPUMembarPass);

  StringRef getArgument() const final { return "test-tritonhcugpu-membar"; }
  StringRef getDescription() const final {
    return "print the result of the membar analysis as run in the hcugpu "
           "backend";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    triton::HCU::annotateLocalLoadsSyncedViaAsyncWait(moduleOp);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    ModuleMembarAnalysis membarPass(&allocation, triton::HCU::membarFilter);
    membarPass.run();
  }
};

} // namespace

namespace mlir::test {
void registerTestHCUGPUMembarPass() {
  PassRegistration<TestHCUGPUMembarPass>();
}
} // namespace mlir::test
