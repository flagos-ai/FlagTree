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

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef intrinsic;
    switch (op.getAxis()) {
    case ProgramIDDim::X:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.x";
      break;
    case ProgramIDDim::Y:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.y";
      break;
    case ProgramIDDim::Z:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.z";
      break;
    default:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.x";
      break;
    }

    auto call =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, {});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateSPMDOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
