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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ThreadIdOpPattern : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
public:
  using ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef intrinsic;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.x";
      break;
    case mlir::gpu::Dimension::y:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.y";
      break;
    case mlir::gpu::Dimension::z:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.z";
      break;
    }

    Type ty = getTypeConverter()->convertType(op.getType());
    auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                                intrinsic, ty, {});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateThreadIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ThreadIdOpPattern>(typeConverter, benefit);
}
