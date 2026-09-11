/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/SignalOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>
#include <limits>

namespace {
using namespace mlir;
namespace tle = mlir::triton::tle;

struct SignalOpConversion : public ConvertOpToLLVMPattern<tle::SignalOp> {
  using ConvertOpToLLVMPattern<tle::SignalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto contextIdx = op.getContextIdx();

    if (contextIdx < 0 || contextIdx > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(op, "invalid context_idx");

#ifdef FLAGCX_ENABLED
    rewriter.replaceOpWithNewOp<tle::FlagCxSignalOp>(
        op, adaptor.getComm(), adaptor.getPeer(), adaptor.getSlotId(),
        adaptor.getValue(), adaptor.getSignalOp(), adaptor.getTeamKind(),
        adaptor.getCoopKind(), contextIdx);
#endif // FLAGCX_ENABLED
    return success();
  }
};

struct SignalWaitOpConversion
    : public ConvertOpToLLVMPattern<tle::SignalWaitOp> {
  SignalWaitOpConversion(LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::SignalWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef FLAGCX_ENABLED
    rewriter.replaceOpWithNewOp<tle::FlagCxSignalWaitOp>(
        op, adaptor.getComm(), adaptor.getSlotId(), adaptor.getWaitKind(),
        adaptor.getTarget(), adaptor.getCoopKind(), adaptor.getContextIdx());
#endif // FLAGCX_ENABLED
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateSignalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SignalOpConversion>(typeConverter, benefit);
  patterns.add<SignalWaitOpConversion>(typeConverter, benefit);
}
