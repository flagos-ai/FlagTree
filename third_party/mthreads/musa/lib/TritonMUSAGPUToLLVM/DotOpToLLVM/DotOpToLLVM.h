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

#ifndef TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H
#define TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace MUSA {

LogicalResult convertWMMADot(triton::musa::WmmaDotOp op,
                             triton::musa::WmmaDotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

LogicalResult convertSQMMADot(triton::musa::SquadDotOp op,
                              triton::musa::SquadDotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId);

LogicalResult convertSQMMADot(triton::mtgpu::SqmmaOp op,
                              triton::mtgpu::SqmmaOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId);

} // namespace MUSA
} // namespace triton
} // namespace mlir

#endif // TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H
