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
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_ALLOCATEPROTONGLOBALSCRATCHBUFFERPASS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"

struct AllocateProtonGlobalScratchBufferPass
    : public impl::AllocateProtonGlobalScratchBufferPassBase<
          AllocateProtonGlobalScratchBufferPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    auto funcOps = triton::proton::gpu::getTritonFunctions(mod);
    assert(funcOps.size() == 1 && "Expected exactly one funcOp");

    int32_t cumulativeMemorySize = 0; // bytes
    std::vector<uint32_t> alignments;

    funcOps[0].walk([&](proton::gpu::GlobalScratchAllocOp op) {
      int offset = llvm::alignTo(cumulativeMemorySize,
                                 proton::gpu::getBytesPerClockEntry());
      op->setAttr("offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      cumulativeMemorySize += op.getNbytes();
      alignments.push_back(op.getAlignment());
    });
    if (alignments.empty())
      return;

    bool allAlignmentsEqual = std::equal(alignments.begin() + 1,
                                         alignments.end(), alignments.begin());
    assert(allAlignmentsEqual &&
           "all global scratch buffer alignment values must be the same");
    mod->setAttr("ttg.profile_scratch_memory_size",
                 builder.getI32IntegerAttr(cumulativeMemorySize));
    mod->setAttr("ttg.profile_scratch_memory_alignment",
                 builder.getI32IntegerAttr(alignments.front()));
  }
};

} // namespace mlir::triton::proton::gpu
