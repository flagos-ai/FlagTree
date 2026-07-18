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
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct MakeTensorPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertOpToLLVMPattern<triton::MakeTensorPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> elems;
    elems.append(adaptor.getOffsets().begin(), adaptor.getOffsets().end());
    elems.append(adaptor.getShape().begin(), adaptor.getShape().end());
    elems.append(adaptor.getStrides().begin(), adaptor.getStrides().end());
    elems.push_back(adaptor.getBase());

    Value packed = ::mlir::packLLElements(op.getLoc(), getTypeConverter(),
                                          elems, rewriter, op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct AdvanceOpConversion : public ConvertOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertOpToLLVMPattern<triton::AdvanceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    auto elems = ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto offsets = adaptor.getOffsets();

    for (auto [i, offset] : llvm::enumerate(offsets)) {
      elems[i] = b.add(offset, elems[i]);
    }

    Value packed = ::mlir::packLLElements(loc, getTypeConverter(), elems,
                                          rewriter, op.getPtr().getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion, AdvanceOpConversion>(typeConverter,
                                                               benefit);
}
