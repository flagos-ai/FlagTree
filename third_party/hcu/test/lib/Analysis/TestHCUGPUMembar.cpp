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
    // Expose actual allocation intervals only for operations marked by a test.
    // This verifies that cross-call tests really reuse the same physical bytes.
    Builder builder(moduleOp.getContext());
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto *funcAllocation = allocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
        if (op->hasAttr("test.check_shared")) {
          // Check the addressed allocation separately from return scratch.
          auto ids = funcAllocation->getBufferIds(op->getOperand(0));
          assert(ids.size() == 1 && "expected one shared-memory target");
          auto id = *ids.begin();
          op->setAttr("test.shared_offset",
                      builder.getI64IntegerAttr(funcAllocation->getOffset(id)));
          op->setAttr(
              "test.shared_size",
              builder.getI64IntegerAttr(funcAllocation->getAllocatedSize(id)));
          op->setAttr("test.has_scratch",
                      builder.getBoolAttr(funcAllocation->getBufferId(op) !=
                                          Allocation::InvalidBufferId));
        }
        if (!op->hasAttr("test.check_allocation"))
          return;
        auto id = funcAllocation->getBufferId(op);
        assert(id != Allocation::InvalidBufferId);
        op->setAttr("test.buffer_offset",
                    builder.getI64IntegerAttr(funcAllocation->getOffset(id)));
        op->setAttr(
            "test.buffer_size",
            builder.getI64IntegerAttr(funcAllocation->getAllocatedSize(id)));
      });
    });
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
