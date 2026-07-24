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

#include "tle/dialect/include/Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"
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

Value getDistDevicePtr(tle::GetDeviceIdOp op, SmallVector<Value> &srcElems) {
  if (!srcElems.empty())
    return srcElems[0];
  else {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    return func.getArgument(1);
  }
}

struct GetDeviceIdOpConversion
    : public ConvertOpToLLVMPattern<tle::GetDeviceIdOp> {
  GetDeviceIdOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetDeviceIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> srcElems;
    if (auto src = adaptor.getInput())
      srcElems = unpackLLElements(loc, src, rewriter);
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      return rewriter.notifyMatchFailure(
          op, "expected parent LLVM::LLVMFuncOp, but none was found. ");
    }
    auto comm = getDistDevicePtr(op, srcElems);
    rewriter.modifyOpInPlace(op, [&]() { op->insertOperands(0, comm); });
    auto localRank = rewriter.create<tle::GetLocalRankOp>(
        op.getLoc(), rewriter.getI32Type(), comm);
    rewriter.replaceOp(op, localRank.getResult());

    return success();
  }
};

} // namespace

void tle::populateGetDeviceIdOpToFlagCxPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetDeviceIdOpConversion>(typeConverter, benefit);
}
