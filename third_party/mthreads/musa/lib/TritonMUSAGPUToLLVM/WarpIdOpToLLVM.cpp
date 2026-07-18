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

#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class WarpIdOpPattern
    : public ConvertOpToLLVMPattern<mlir::triton::gpu::WarpIdOp> {
public:
  using ConvertOpToLLVMPattern<
      mlir::triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // This is runtime-constant for a program instance; move it to function
    // entry unless we are inside a warp-specialized partition.
    std::optional<int> startWarpId = getWarpGroupStartWarpId(op->getBlock());
    if (!startWarpId) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      rewriter.setInsertionPoint(
          &funcOp.getFunctionBody().getBlocks().front().front());
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tid = LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.musa.read.ptx.sreg.tid.x", i32_ty, {})
                    .getResult(0);
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpId = b.udiv(tid, b.i32_val(threadsPerWarp));

    if (startWarpId)
      warpId = b.sub(warpId, b.i32_val(*startWarpId));

    rewriter.replaceOp(op, warpId);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, benefit);
}
