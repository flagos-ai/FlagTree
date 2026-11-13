/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

#ifndef TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H
#define TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H

#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace HoistBroadcast {
using namespace mlir;
using namespace triton;

class BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BroadcastHoister {
public:
  BroadcastHoister(triton::BroadcastOp op);
  LogicalResult parse(Value operand, const Location &loc,
                      ConversionPatternRewriter &rewriter);
  LogicalResult parseAddptr(triton::AddPtrOp op, const Location &loc,
                            ConversionPatternRewriter &rewriter);
  LogicalResult parseBroadcast(triton::BroadcastOp op, const Location &loc,
                               ConversionPatternRewriter &rewriter);
  LogicalResult parseSplat(triton::SplatOp op, const Location &loc,
                           ConversionPatternRewriter &rewriter);
  LogicalResult findSrc(Value operand);
  LogicalResult replaceBroadcastOp(triton::BroadcastOp op,
                                   ConversionPatternRewriter &rewriter);
  bool canBroadcast();

private:
  Value source;
  triton::BroadcastOp opToHoist;
  SmallVector<int64_t> tensorSizes;
  llvm::SmallDenseMap<Value, Value> broadcastMap;
};
} // namespace HoistBroadcast

#endif // TRITON_ADAPTER_TRITONTOLINALG_HOISTBROADCAST_H
