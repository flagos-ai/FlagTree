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

#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only

namespace {

using namespace mlir;
using namespace mlir::triton;

struct XPUGetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value retVal = rewriter.create<mlir::LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(ctx), i32_val(1));

    rewriter.replaceOp(op, {retVal});
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<XPUGetNumProgramsOpConversion>(typeConverter, benefit);
}
