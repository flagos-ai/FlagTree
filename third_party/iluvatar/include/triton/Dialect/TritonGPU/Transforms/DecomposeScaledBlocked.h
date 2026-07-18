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

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {
public:
  DecomposeScaledBlocked(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override;

protected:
  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const;
  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const;
  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 ModuleOp mod, TypedValue<RankedTensorType> scale,
                 int dim) const;
  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const;
  virtual TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                                DotScaledOp scaledDotOp,
                                                int opIdx,
                                                FloatType computeType) const;
  TypedValue<RankedTensorType>
  cvtDotOperand(PatternRewriter &rewriter, DotScaledOp scaledDotOp, int opIdx,
                TypedValue<RankedTensorType> v) const;
  TypedValue<RankedTensorType>
  extendAndBroadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                          TypedValue<RankedTensorType> &scale,
                          FloatType computeType, RankedTensorType dstType,
                          int opIdx) const;
  static SmallVector<int, 2> getTransposeOrder(int rank);
};

void populateDecomposeScaledBlockedPatterns(mlir::RewritePatternSet &patterns,
                                            int benefit);

} // namespace mlir::triton::gpu
