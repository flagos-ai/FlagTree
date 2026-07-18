// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
