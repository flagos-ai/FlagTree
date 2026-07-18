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

#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/PatternTritonHCUGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/hcu/lib/TritonHCUGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using mlir::LLVM::HCU::upcast4xMxfp8_HW;
using mlir::LLVM::HCU::upcast8xMxfp4_HW;

namespace {
struct ScaledUpcastFp4OpPattern
    : ConvertOpToLLVMPattern<hcugpu::ScaledUpcastFp4Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hcugpu::ScaledUpcastFp4Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    SmallVector<Value> results;
    results.reserve(inputVals.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < inputVals.size(); i += 4) {
      SmallVector<Value, 4> v4i32 =
          elemType.isF16() ? upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkF16Fp4Op>(
                                 rewriter, loc, inputVals, i, scaleVals[i * 2],
                                 /*useShiftedScale=*/true)
                           : upcast8xMxfp4_HW<ROCDL::CvtScaleF32PkBf16Fp4Op>(
                                 rewriter, loc, inputVals, i, scaleVals[i * 2],
                                 /*useShiftedScale=*/true);
      for (int j = 0; j < 4; j++) {
        Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }
};

struct ScaledUpcastFp8OpPattern
    : ConvertOpToLLVMPattern<hcugpu::ScaledUpcastFp8Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hcugpu::ScaledUpcastFp8Op upcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = upcastOp.getLoc();
    auto elemType = upcastOp.getType().getElementType();
    auto fp8ElemType = upcastOp.getInput().getType().getElementType();

    auto inputVals = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);

    assert(inputVals.size() % 4 == 0);
    assert(inputVals.size() == scaleVals.size());
    SmallVector<Value> results;
    results.reserve(inputVals.size());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < inputVals.size(); i += 4) {
      SmallVector<Value, 2> v2i32 =
          elemType.isF16()
              ? (isa<Float8E4M3FNType>(fp8ElemType)
                     ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Fp8Op>(
                           rewriter, loc, inputVals, i, scaleVals[i],
                           /*useShiftedScale=*/true)
                     : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkF16Bf8Op>(
                           rewriter, loc, inputVals, i, scaleVals[i],
                           /*useShiftedScale=*/true))
              : (isa<Float8E4M3FNType>(fp8ElemType)
                     ? upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Fp8Op>(
                           rewriter, loc, inputVals, i, scaleVals[i],
                           /*useShiftedScale=*/true)
                     : upcast4xMxfp8_HW<ROCDL::CvtScaleF32PkBf16Bf8Op>(
                           rewriter, loc, inputVals, i, scaleVals[i],
                           /*useShiftedScale=*/true));
      for (int j = 0; j < 2; j++) {
        Value elements = b.bitcast(v2i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  upcastOp.getType());
    rewriter.replaceOp(upcastOp, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::HCU::populateScaledUpcastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ScaledUpcastFp4OpPattern>(typeConverter, benefit);
  patterns.add<ScaledUpcastFp8OpPattern>(typeConverter, benefit);
}
