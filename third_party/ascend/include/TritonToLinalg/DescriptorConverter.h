/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DESCRIPTORCONVERTER_H
#define TRITON_ADAPTER_DESCRIPTORCONVERTER_H

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

struct Descriptor {
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  triton::PaddingOptionAttr padding;
};

bool hasATensorDescriptorType(mlir::TypeRange types);

class DescriptorLoadConverter
    : public OpConversionPattern<triton::DescriptorLoadOp> {
public:
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorStoreConverter
    : public OpConversionPattern<triton::DescriptorStoreOp> {
public:
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorGatherConverter
    : public OpConversionPattern<triton::DescriptorGatherOp> {
public:
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorScatterConverter
    : public OpConversionPattern<triton::DescriptorScatterOp> {
public:
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorReduceConverter
    : public OpConversionPattern<triton::DescriptorReduceOp> {
public:
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // end of namespace DescriptorConverter

#endif // TRITON_ADAPTER_DESCRIPTORCONVERTER_H
