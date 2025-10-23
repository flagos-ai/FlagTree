#pragma once

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_BUBBLEUPOPERATION
#include "../../include/TritonToUnstructure/Passes.h.inc"

#define GEN_PASS_DEF_BUBBLEUPOPERATION
#include "../../include/TritonToUnstructure/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createBubbleUpOperationPass(const BubbleUpOperationOptions &options = {});

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

class BubbleUpExtract : public OpRewritePattern<tensor::ExtractOp> {
public:
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  explicit BubbleUpExtract(MLIRContext *context, bool enableAggressiveMode);

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override;

private:
  Value createExtractOp(Value value, ArrayRef<Value> indices, Location loc,
                        PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpIntBinaryOp(Operation *op, BinOpTy binOp,
                           ArrayRef<Value> indices, Location loc,
                           PatternRewriter &rewriter) const;
  template <typename BinOpTy>
  void bubbleUpFloatBinaryOp(Operation *op, BinOpTy binOp,
                             ArrayRef<Value> indices, Location loc,
                             PatternRewriter &rewriter) const;
  template <typename ParentOpTy>
  void bubbleUpOperation(Operation *op, ParentOpTy parentOp,
                         ArrayRef<Value> indices, Location loc,
                         PatternRewriter &rewriter) const = delete;

  template <>
  void bubbleUpOperation<arith::ExtSIOp>(Operation *op, arith::ExtSIOp parentOp,
                                         ArrayRef<Value> indices, Location loc,
                                         PatternRewriter &rewriter) const;

  template <>
  void bubbleUpOperation<arith::CmpIOp>(Operation *op, arith::CmpIOp parentOp,
                                        ArrayRef<Value> indices, Location loc,
                                        PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<arith::TruncFOp>(Operation *op,
                                          arith::TruncFOp parentOp,
                                          ArrayRef<Value> indices, Location loc,
                                          PatternRewriter &rewriter) const;

  template <>
  void bubbleUpOperation<arith::ExtFOp>(Operation *op,
                                        arith::ExtFOp parentOp,
                                        ArrayRef<Value> indices, Location loc,
                                        PatternRewriter &rewriter) const;

  template <>
  void bubbleUpOperation<arith::FPToSIOp>(Operation *op,
                                          arith::FPToSIOp parentOp,
                                          ArrayRef<Value> indices, Location loc,
                                          PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<arith::SIToFPOp>(Operation *op,
                                          arith::SIToFPOp parentOp,
                                          ArrayRef<Value> indices, Location loc,
                                          PatternRewriter &rewriter) const;
  template <>
  void
  bubbleUpOperation<triton::ClampFOp>(Operation *op, triton::ClampFOp parentOp,
                                      ArrayRef<Value> indices, Location loc,
                                      PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<arith::CmpFOp>(Operation *op, arith::CmpFOp parentOp,
                                        ArrayRef<Value> indices, Location loc,
                                        PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<triton::BroadcastOp>(Operation *op,
                                              triton::BroadcastOp parentOp,
                                              ArrayRef<Value> indices,
                                              Location loc,
                                              PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<triton::ExpandDimsOp>(Operation *op,
                                               triton::ExpandDimsOp parentOp,
                                               ArrayRef<Value> indices,
                                               Location loc,
                                               PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<triton::SplatOp>(Operation *op,
                                          triton::SplatOp parentOp,
                                          ArrayRef<Value> indices, Location loc,
                                          PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<triton::MakeRangeOp>(Operation *op,
                                              triton::MakeRangeOp parentOp,
                                              ArrayRef<Value> indices,
                                              Location loc,
                                              PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<math::FloorOp>(Operation *op, math::FloorOp parentOp,
                                        ArrayRef<Value> indices, Location loc,
                                        PatternRewriter &rewriter) const;
  template <>
  void bubbleUpOperation<math::CeilOp>(Operation *op, math::CeilOp parentOp,
                                       ArrayRef<Value> indices, Location loc,
                                       PatternRewriter &rewriter) const;

  bool enableAggressiveMode;
};

class BubbleUpOperationPass
    : public ::impl::BubbleUpOperationBase<BubbleUpOperationPass> {
public:
  explicit BubbleUpOperationPass(const BubbleUpOperationOptions &options);
  void runOnOperation() override;
};