#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonXPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEXPUSHAREDMEMORY
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

unsigned getXPUAllocationAnalysisScratchSize(Operation *op) {
  if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(op)) {
    auto srcTy = cast<RankedTensorType>(reduceOp.getOperands()[0].getType());
    auto smemShape = convertType<unsigned>(srcTy.getShape());
    smemShape[reduceOp.getAxis()] = 64;

    unsigned bytesPerElem = 0;
    for (const auto &ty : reduceOp.getElementTypes()) {
      bytesPerElem +=
          ceil<unsigned>(getElementTypeOrSelf(ty).getIntOrFloatBitWidth(), 8);
    }
    return bytesPerElem * product<unsigned>(smemShape);
  }

  if (isa<triton::xpu::ScanOp>(op))
    return 128 * 4;

  return triton::defaultAllocationAnalysisScratchSizeFn(op);
}

struct AllocateXPUSharedMemory
    : public triton::impl::AllocateXPUSharedMemoryBase<
          AllocateXPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod, getXPUAllocationAnalysisScratchSize);

    triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateXPUSharedMemoryPass() {
  return std::make_unique<AllocateXPUSharedMemory>();
}

} // namespace mlir::triton
