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

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"

using namespace mlir;

namespace {

struct WGMMASharedOperandFenceOpConversion
    : public ConvertOpToLLVMPattern<triton::tle::WGMMASharedOperandFenceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::tle::WGMMASharedOperandFenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = NVVM::ProxyKind::async_shared;
    auto space = op.getBCluster() ? NVVM::SharedSpace::shared_cluster
                                  : NVVM::SharedSpace::shared_cta;
    auto ctx = rewriter.getContext();
    auto spaceAttr = NVVM::SharedSpaceAttr::get(ctx, space);
    rewriter.replaceOpWithNewOp<NVVM::FenceProxyOp>(op, kind, spaceAttr);
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateWGMMASharedOperandFenceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<WGMMASharedOperandFenceOpConversion>(typeConverter, benefit);
}
