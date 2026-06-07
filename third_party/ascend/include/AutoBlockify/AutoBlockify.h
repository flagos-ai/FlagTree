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

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_AUTOBLOCKIFY
#include "ascend/include/AutoBlockify/Passes.h.inc"

#define GEN_PASS_DEF_AUTOBLOCKIFY
#include "ascend/include/AutoBlockify/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createAutoBlockifyPass(const AutoBlockifyOptions &options = {});

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class PropagateUnrealizedCastDown
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit PropagateUnrealizedCastDown(MLIRContext *context,
                                       Value logicalBlockId,
                                       Value logicalBlockNum,
                                       int autoBlockifySize);

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override;

private:
  void handleBlockifyLoop(scf::ForOp blockifyLoop, Operation *op, PatternRewriter &rewriter) const;
  void rewriteSplat(UnrealizedConversionCastOp op, triton::SplatOp splatOp,
                    PatternRewriter &rewriter) const;
  void rewriteExpandDims(UnrealizedConversionCastOp op,
                         triton::ExpandDimsOp expandDimsOp,
                         PatternRewriter &rewriter) const;
  void rewriteReduce(UnrealizedConversionCastOp op, triton::ReduceOp reduceOp,
                     PatternRewriter &rewriter) const;
  void rewriteScan(UnrealizedConversionCastOp op, triton::ScanOp scanOp,
                   PatternRewriter &rewriter) const;
  void rewriteLoad(UnrealizedConversionCastOp op, triton::LoadOp loadOp,
                   PatternRewriter &rewriter) const;
  void rewriteStore(UnrealizedConversionCastOp op, triton::StoreOp storeOp,
                    PatternRewriter &rewriter) const;
  void rewriteAtomicRMW(UnrealizedConversionCastOp op,
                        triton::AtomicRMWOp atomicRMWOp,
                        PatternRewriter &rewriter) const;
  void rewriteAssert(UnrealizedConversionCastOp op, triton::AssertOp assertOp,
                     PatternRewriter &rewriter) const;
  void rewriteExtractSlice(UnrealizedConversionCastOp op,
                           tensor::ExtractSliceOp extractSliceOp,
                           PatternRewriter &rewriter) const;
  void rewriteInsertSlice(UnrealizedConversionCastOp op,
                          tensor::InsertSliceOp insertSliceOp,
                          PatternRewriter &rewriter) const;
  void rewriteWhile(UnrealizedConversionCastOp op, scf::WhileOp whileOp,
                    PatternRewriter &rewriter) const;
  void rewriteLoop(UnrealizedConversionCastOp op, LoopLikeOpInterface loopOp,
                   PatternRewriter &rewriter) const;
  void rewriteIf(UnrealizedConversionCastOp &op, scf::IfOp ifOp, ArrayRef<int64_t> indices,
                 PatternRewriter &rewriter) const;
  void rewriteYield(UnrealizedConversionCastOp &op, scf::YieldOp yieldOp,
                    PatternRewriter &rewriter) const;
  void rewriteCondition(UnrealizedConversionCastOp op,
                        scf::ConditionOp conditionOp,
                        PatternRewriter &rewriter) const;
  void rewriteGeneraleOp(UnrealizedConversionCastOp op, Operation *generalOp,
                         PatternRewriter &rewriter) const;

  Value logicalBlockId;
  Value logicalBlockNum;
  int autoBlockifySize;
};

class AutoBlockifyPass : public ::impl::AutoBlockifyBase<AutoBlockifyPass> {
public:
  explicit AutoBlockifyPass(const AutoBlockifyOptions &options);
  void runOnOperation() override;

private:
  bool checkBlockifiable(Value v);
  void preProcess(triton::FuncOp func);

  DenseSet<Value> checkedValues;
  Value logicalBlockId;
  Value logicalBlockNum;
};
