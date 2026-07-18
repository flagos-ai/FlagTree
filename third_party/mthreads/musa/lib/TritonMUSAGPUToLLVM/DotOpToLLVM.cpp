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

#include "DotOpToLLVM/DotOpToLLVM.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto lhsTy = cast<RankedTensorType>(op.getA().getType());
    auto rhsTy = cast<RankedTensorType>(op.getB().getType());
    if ((op.getInputPrecision() == triton::InputPrecision::BF16x3 ||
         op.getInputPrecision() == triton::InputPrecision::BF16x6) &&
        lhsTy.getElementType().isF32() && rhsTy.getElementType().isF32()) {
      return op.emitError(
          "bf16x3/bf16x6 tt.dot must be rewritten by TritonGPUF32DotTC "
          "before MUSA LLVM lowering");
    }
    if (isa<MUSAWmmaEncodingAttr, MUSASqmmaEncodingAttr>(
            resultTy.getEncoding()))
      return op.emitError("MUSA matmul with mma encoding must be rewritten to "
                          "ttmg.wmma_dot/ttmg.squad_dot before LLVM lowering");
    if (isa<BlockedEncodingAttr>(resultTy.getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported MUSA DotOp encoding in DotOp lowering.");
  }
};

} // namespace

void mlir::triton::MUSA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
