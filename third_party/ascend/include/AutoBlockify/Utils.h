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

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

constexpr llvm::StringLiteral autoBlockifySizeAttr = "auto_blockify_size";
constexpr llvm::StringLiteral logicalBlockIdAttr = "logical_block_id";
constexpr llvm::StringLiteral autoBlockifyLoopAttr = "auto_blockify_loop";
constexpr llvm::StringLiteral autoBlockifyRegionOpAttr =
    "auto_blockify_region_op";

RankedTensorType getExpandedType(Type type, UnrealizedConversionCastOp op);

Value rewriteValue(Value value, UnrealizedConversionCastOp op,
                   OpBuilder &builder);

void replaceValue(Operation *newOp, Operation *oldOp, Value newMask,
                  RewriterBase &rewriter,
                  ArrayRef<int64_t> replaceIndices = {});

Value createMask(Value mask, Value uccMask, ArrayRef<int64_t> targetShape,
                 RewriterBase &rewriter);

void mapRegionIterArg(IRMapping &mapping, ValueRange oldArgs,
                      ValueRange newArgs, ArrayRef<int64_t> indices, Value mask,
                      OpBuilder &builder);

void mapYieldedValue(IRMapping &mapping, scf::YieldOp yieldOp,
                     ArrayRef<int64_t> indices, UnrealizedConversionCastOp op,
                     OpBuilder &builder);

Operation *createBlockifyLoop(Operation *targetOp,
                              UnrealizedConversionCastOp op,
                              Value logicalBlockId, Value logicalBlockNum,
                              int autoBlockifySize, RewriterBase &rewriter);

std::optional<scf::ForOp> getBlockifyLoop(Operation *op);
