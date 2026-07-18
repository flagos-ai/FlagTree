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

#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_ALLOCATEPROTONSHAREDMEMORYPASS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"

struct AllocateProtonSharedMemoryPass
    : public impl::AllocateProtonSharedMemoryPassBase<
          AllocateProtonSharedMemoryPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    int sharedMemUsed = 0;
    if (mod->hasAttr("ttg.shared"))
      sharedMemUsed =
          mod->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    assert(llvm::range_size(mod.getOps<triton::FuncOp>()) == 1);
    FuncOp func = *mod.getOps<triton::FuncOp>().begin();

    int totalSharedMemSize = 0;
    int count = 0;
    func.walk([&](triton::gpu::LocalAllocOp alloc) {
      // We ignore the shared memory allocations that have been allocated by the
      // triton conversion pass.
      if (!alloc->hasAttr("allocation.offset")) {
        int offset =
            llvm::alignTo(sharedMemUsed, proton::gpu::getBytesPerClockEntry());
        alloc->setAttr("allocation.offset",
                       IntegerAttr::get(IntegerType::get(ctx, 32), offset));
        // Compute the proton buffer size in bytes.
        auto memDescTy =
            mlir::cast<triton::gpu::MemDescType>(alloc.getResult().getType());
        int bufferSizeInBytes =
            mlir::ShapedType::getNumElements(memDescTy.getShape()) *
            memDescTy.getElementType().getIntOrFloatBitWidth() / 8;

        totalSharedMemSize = offset + bufferSizeInBytes;
        count++;
      }
    });

    if (count == 0) {
      totalSharedMemSize = sharedMemUsed;
    }

    mod->setAttr("ttg.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        totalSharedMemSize));
  }
};

} // namespace mlir::triton::proton::gpu
