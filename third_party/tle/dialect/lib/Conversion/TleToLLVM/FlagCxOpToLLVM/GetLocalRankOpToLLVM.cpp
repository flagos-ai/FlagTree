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

#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

struct GetNumPesOpConversion : public ConvertOpToLLVMPattern<tle::GetNumPesOp> {
  GetNumPesOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetNumPesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[GetNumPesOpConversion] " << msg << "\n";
      return failure();
    };
    auto loc = op.getLoc();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto getLocalPeCall = tle::getNumPesFunCall(loc, rewriter, srcElems[0]);

    Value nPes = getLocalPeCall.getResult();
    if (!nPes.getType().isInteger(32))
      return reportFailure("expected i32 result");
    rewriter.replaceOp(op, nPes);
    return success();
  }
};

struct GetLocalRankOpConversion
    : public ConvertOpToLLVMPattern<tle::GetLocalRankOp> {
  GetLocalRankOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetLocalRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[GetLocalRankOpConversion] " << msg << "\n";
      return failure();
    };
    auto loc = op.getLoc();
    auto comm = op.getSrc();
    auto getLocalPeCall = tle::getLocalPeFuncCall(loc, rewriter, comm);

    Value localPe = getLocalPeCall.getResult();
    if (!localPe.getType().isInteger(32))
      return reportFailure("expected i32 result");
    rewriter.replaceOp(op, localPe);
    return success();
  }
};

} // namespace
void tle::populateGetLocalRankOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               PatternBenefit benefit) {
  patterns.add<GetLocalRankOpConversion>(typeConverter, benefit);
}

void tle::populateGetNumPesOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<GetNumPesOpConversion>(typeConverter, benefit);
}
