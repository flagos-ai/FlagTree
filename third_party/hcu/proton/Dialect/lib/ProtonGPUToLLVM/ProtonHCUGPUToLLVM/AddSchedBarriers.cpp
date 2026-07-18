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

#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton::gpu {
#define GEN_PASS_DEF_ADDSCHEDBARRIERS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton::gpu
} // namespace mlir

namespace {

struct AddSchedBarriers
    : public mlir::triton::proton::gpu::impl::AddSchedBarriersBase<
          AddSchedBarriers> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    auto funcOps = triton::proton::gpu::getTritonFunctions(mod);
    assert(funcOps.size() == 1 && "Expected exactly one funcOp");

    IntegerAttr zeroAttrValue =
        builder.getI32IntegerAttr(static_cast<int32_t>(0));

    funcOps[0].walk([&](mlir::triton::proton::gpu::ReadCounterOp op) {
      auto loc = op.getLoc();
      if (!isa_and_nonnull<ROCDL::SchedBarrier>(op->getPrevNode())) {
        builder.setInsertionPoint(op);
        ROCDL::SchedBarrier::create(builder, loc, zeroAttrValue);
      }
    });

    funcOps[0].walk([&](mlir::triton::proton::gpu::CircularStoreOp op) {
      auto loc = op.getLoc();
      if (!isa_and_nonnull<ROCDL::SchedBarrier>(op->getNextNode())) {
        builder.setInsertionPointAfter(op);
        ROCDL::SchedBarrier::create(builder, loc, zeroAttrValue);
      }
    });
  }
};

} // namespace

namespace mlir::triton::proton::gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAddSchedBarriersPass() {
  return std::make_unique<AddSchedBarriers>();
}

} // namespace mlir::triton::proton::gpu
