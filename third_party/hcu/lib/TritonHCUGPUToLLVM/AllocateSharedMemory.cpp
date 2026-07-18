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

#include "Analysis/HCUGPUAllocation.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::HCU;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEHCUGPUSHAREDMEMORY
#include "TritonHCUGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct AllocateHCUGPUSharedMemory
    : public mlir::triton::impl::AllocateHCUGPUSharedMemoryBase<
          AllocateHCUGPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod, HCUAllocationAnalysisScratchSizeFn);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace
