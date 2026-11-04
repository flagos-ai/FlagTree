/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxBarrierOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"

namespace {
using namespace mlir;
using namespace mlir::triton;

struct FlagCxBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::FlagCxBarrierOp> {
  FlagCxBarrierOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::FlagCxBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    tle::getUnifiedBarrierFuncCall(
        op.getLoc(), rewriter, adaptor.getComm(),
        static_cast<int32_t>(op.getTeamKind()),
        static_cast<int32_t>(op.getIndexAttr().getInt()),
        static_cast<int32_t>(op.getContextIdAttr().getInt()),
        static_cast<int32_t>(op.getCoopKind()),
        static_cast<int32_t>(op.getOrder()),
        static_cast<int32_t>(op.getScope()),
        op.getBarrierTypeAttr().getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DeviceIntraBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DeviceIntraBarrierOp> {
  DeviceIntraBarrierOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::DeviceIntraBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto coopValue = op.getCoopKindAttr().getInt();
    auto orderValue = op.getOrderAttr().getInt();
    auto barrierType = op.getBarrierTypeAttr().getValue();
    if (!llvm::is_contained(std::array<size_t, 5>{0, 1, 2, 3, 4}, coopValue))
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");

    if (!llvm::is_contained(std::array<size_t, 4>{0, 1, 2, 3}, orderValue))
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");

    if (auto index = adaptor.getDynamicIndex())
      tle::getBarrierFuncCall(loc, rewriter, adaptor.getComm(), index,
                              coopValue, orderValue, barrierType);
    else
      tle::getBarrierFuncCall(loc, rewriter, adaptor.getComm(),
                              op.getIndexAttr().getInt(), coopValue, orderValue,
                              barrierType);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateFlagCxBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCxBarrierOpConversion, DeviceIntraBarrierOpConversion>(
      typeConverter, benefit);
}
