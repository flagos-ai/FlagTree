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
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MACAMmaEncodingAttr;

LogicalResult convertMMAMACA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MACAMmaEncodingAttr mmaLayout = dyn_cast<MACAMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (!isOuter && mmaLayout &&
        supportMMA(op, mmaLayout.getVersionMajor(),
                   mmaLayout.getVersionMinor())) {
      return convertMMAMACA(op, adaptor, getTypeConverter(), rewriter);
      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

void mlir::triton::METAX::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
