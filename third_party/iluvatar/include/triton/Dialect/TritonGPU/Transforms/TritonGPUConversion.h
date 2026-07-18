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

//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonGPUTypeConverter : public TypeConverter {
public:
  TritonGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                         int numCTAs, bool enableSourceRemat);
  int getNumWarps() const { return numWarps; }
#ifdef __ILUVATAR_TLE__
  int getNumWarps(Value value) const;
#endif
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }
#ifdef __ILUVATAR_TLE__
  RankedTensorType convertRankedTensorType(RankedTensorType type,
                                           int contextualNumWarps) const;
#endif

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

class TritonGPUConversionTarget : public ConversionTarget {
public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx,
                                     TritonGPUTypeConverter &typeConverter);

  // Determine whether the operation is currently legal. I.e. it has layouts
  // assigned to its tensor operands and results.
  static bool isDynamicallyLegal(Operation *op,
                                 const TypeConverter &typeConverter);
};

namespace impl {
LogicalResult convertGatherScatterOp(Operation *op, ValueRange operands,
                                     OpOperand &xOffsetsMutable,
                                     const TypeConverter &typeConverter,
                                     ConversionPatternRewriter &rewriter);
} // namespace impl

// Generic pattern for converting a TMA gather or scatter operation.
template <typename OpT>
struct GatherScatterOpPattern : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return impl::convertGatherScatterOp(op, adaptor.getOperands(),
                                        op.getXOffsetsMutable(),
                                        *this->getTypeConverter(), rewriter);
  }
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
