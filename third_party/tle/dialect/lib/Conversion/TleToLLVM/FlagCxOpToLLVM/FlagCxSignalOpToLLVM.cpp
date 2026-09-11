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

#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxSignalOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>
#include <limits>

namespace {
using namespace mlir;
namespace tle = mlir::triton::tle;

struct FlagCxSignalOpConversion
    : public ConvertOpToLLVMPattern<tle::FlagCxSignalOp> {
  using ConvertOpToLLVMPattern<tle::FlagCxSignalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::FlagCxSignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    uint32_t contextIdx = op.getContextIdx();

    if (contextIdx > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(op, "invalid context_idx");

    tle::getSignalFuncCall(loc, rewriter, adaptor.getComm(), adaptor.getPeer(),
                           adaptor.getSlotId(), adaptor.getValue(), contextIdx,
                           adaptor.getTeamKind(), adaptor.getCoopKind(),
                           adaptor.getSignalOp());
    rewriter.eraseOp(op);
    return success();
  }
};

struct FlagCxSignalWaitOpConversion
    : public ConvertOpToLLVMPattern<tle::FlagCxSignalWaitOp> {
  FlagCxSignalWaitOpConversion(LLVMTypeConverter &typeConverter,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::FlagCxSignalWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto comm = adaptor.getComm();
    auto wait_kind = adaptor.getWaitKind();
    auto coop_kind = adaptor.getCoopKind();
    auto slot_id = adaptor.getSlotId();
    auto target = adaptor.getTarget();
    auto context_idx = adaptor.getContextIdx();

    tle::getDevNetWaitFuncCallByKind(loc, rewriter, comm, slot_id, wait_kind,
                                     target, coop_kind, context_idx);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateFlagCxSignalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCxSignalOpConversion>(typeConverter, benefit);
  patterns.add<FlagCxSignalWaitOpConversion>(typeConverter, benefit);
}
