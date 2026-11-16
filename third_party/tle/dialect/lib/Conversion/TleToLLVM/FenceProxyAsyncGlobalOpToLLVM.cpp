// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"

using namespace mlir;

namespace {

struct FenceProxyAsyncGlobalOpConversion
    : ConvertOpToLLVMPattern<triton::tle::FenceProxyAsyncGlobalOp> {
  FenceProxyAsyncGlobalOpConversion(LLVMTypeConverter &converter,
                                   int computeCapability, int ptxVersion,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        computeCapability(computeCapability), ptxVersion(ptxVersion) {}

  LogicalResult
  matchAndRewrite(triton::tle::FenceProxyAsyncGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (computeCapability < 90 || ptxVersion < 80)
      return op.emitError("requires SM90 or newer and PTX ISA 8.0 or newer");
    rewriter.replaceOpWithNewOp<NVVM::FenceProxyOp>(
        op, NVVM::ProxyKind::async_global, NVVM::SharedSpaceAttr{});
    return success();
  }

  int computeCapability;
  int ptxVersion;
};

} // namespace

void mlir::triton::tle::populateFenceProxyAsyncGlobalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, int ptxVersion, unsigned benefit) {
  patterns.add<FenceProxyAsyncGlobalOpConversion>(
      typeConverter, computeCapability, ptxVersion, benefit);
}
