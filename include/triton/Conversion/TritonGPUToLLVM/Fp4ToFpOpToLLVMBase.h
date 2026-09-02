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

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <array>

namespace mlir::triton::gpu {

class Fp4ToFpOpConversionBase : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpConversionBase(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit);

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

protected:
  /// Backend-specific implementation that unpacks a 4 element packed vector of
  /// fp4x2 into 8 elemements of \p elemType.
  virtual std::array<Value, 8>
  upcastPackedFp4(Fp4ToFpOp op, ConversionPatternRewriter &rewriter,
                  Value packedVec, Type elemType) const = 0;
};

} // namespace mlir::triton::gpu

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_FP4_TO_FP_OP_TO_LLVM_BASE_H
