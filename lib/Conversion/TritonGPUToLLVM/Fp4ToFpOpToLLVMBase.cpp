/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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

#include "triton/Conversion/TritonGPUToLLVM/Fp4ToFpOpToLLVMBase.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::gpu {

Fp4ToFpOpConversionBase::Fp4ToFpOpConversionBase(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

LogicalResult Fp4ToFpOpConversionBase::matchAndRewrite(
    Fp4ToFpOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto resTy = op.getType();
  auto elemType = resTy.getElementType();
  assert(elemType == f16_ty || elemType == bf16_ty);

  auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

  SmallVector<Value> results;
  results.reserve(xVals.size() * 2);
  assert(xVals.size() % 4 == 0);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (int i = 0; i < xVals.size(); i += 4) {
    Value packedVec = b.undef(vec_ty(i8_ty, 4));
    for (int j = 0; j < 4; ++j)
      packedVec = b.insert_element(packedVec, xVals[i + j], b.i32_val(j));
    auto upcast = upcastPackedFp4(op, rewriter, packedVec, elemType);
    results.append(upcast.begin(), upcast.end());
  }

  Value result =
      packLLElements(loc, getTypeConverter(), results, rewriter, resTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::gpu
