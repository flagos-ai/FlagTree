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

#define GEN_PASS_DECL_BUBBLEUPOPERATION
#include "ascend/include/TritonToUnstructure/Passes.h.inc"

#define GEN_PASS_DEF_BUBBLEUPOPERATION
#include "ascend/include/TritonToUnstructure/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createBubbleUpOperationPass(const BubbleUpOperationOptions &options = {});

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

template <typename ExtractOpTy>
class BubbleUpExtract : public OpRewritePattern<ExtractOpTy> {
  static_assert(std::is_same_v<ExtractOpTy, tensor::ExtractOp> ||
                std::is_same_v<ExtractOpTy, tensor::ExtractSliceOp>);

public:
  using OpRewritePattern<ExtractOpTy>::OpRewritePattern;

  explicit BubbleUpExtract(MLIRContext *context, bool enableAggressiveMode);

  LogicalResult matchAndRewrite(ExtractOpTy op,
                                PatternRewriter &rewriter) const override;

private:
  Value createExtractOp(ExtractOpTy op, Value value, Location loc,
                        PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpIntBinaryOp(ExtractOpTy op, BinOpTy binOp, Location loc,
                           PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpFloatBinaryOp(ExtractOpTy op, BinOpTy binOp, Location loc,
                             PatternRewriter &rewriter) const;

  void bubbleUpOperation(ExtractOpTy op, arith::ExtSIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::CmpIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  // Pushes extract(select(condition, lhs, rhs)) through the select. A shaped
  // condition is extracted at the same position while a scalar condition is
  // reused directly.
  void bubbleUpOperation(ExtractOpTy op, arith::SelectOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::TruncFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::ExtFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::FPToSIOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::SIToFPOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::ClampFOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, arith::CmpFOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::BroadcastOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::ExpandDimsOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::SplatOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::MakeRangeOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, triton::AddPtrOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, math::FloorOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, math::CeilOp parentOp, Location loc,
                         PatternRewriter &rewriter) const;
  void bubbleUpOperation(ExtractOpTy op, tensor::ExtractSliceOp parentOp,
                         Location loc, PatternRewriter &rewriter) const;

  bool enableAggressiveMode;
};

class BubbleUpOperationPass
    : public ::impl::BubbleUpOperationBase<BubbleUpOperationPass> {
public:
  explicit BubbleUpOperationPass(const BubbleUpOperationOptions &options);
  void runOnOperation() override;
};
