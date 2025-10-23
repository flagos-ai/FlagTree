#include "TritonToUnstructure/BubbleUpOperation.h"
#include "Utils/Utils.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "triton-bubble-up-operation"

BubbleUpExtract::BubbleUpExtract(MLIRContext *context,
                                 bool enableAggressiveMode)
    : OpRewritePattern<tensor::ExtractOp>(context),
      enableAggressiveMode(enableAggressiveMode) {}

LogicalResult
BubbleUpExtract::matchAndRewrite(tensor::ExtractOp op,
                                 PatternRewriter &rewriter) const {
  auto tensorValue = op.getTensor();
  auto parentOp = tensorValue.getDefiningOp();
  auto indices =
      SmallVector<Value>(op.getIndices().begin(), op.getIndices().end());
  auto loc = op.getLoc();

  if (!parentOp ||
      (!enableAggressiveMode && !parentOp->hasOneUse())) {
    return failure();
  }

  if (auto extsiOp = dyn_cast<arith::ExtSIOp>(parentOp)) {
    bubbleUpOperation(op, extsiOp, indices, loc, rewriter);
  } else if (auto addIOp = dyn_cast<arith::AddIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, addIOp, indices, loc, rewriter);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, subIOp, indices, loc, rewriter);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, mulIOp, indices, loc, rewriter);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, divSIOp, indices, loc, rewriter);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, remSIOp, indices, loc, rewriter);
  } else if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, maxSIOp, indices, loc, rewriter);
  } else if (auto minSIOp = dyn_cast<arith::MinSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, minSIOp, indices, loc, rewriter);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, andIOp, indices, loc, rewriter);
  } else if (auto orIOp = dyn_cast<arith::OrIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, orIOp, indices, loc, rewriter);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(parentOp)) {
    bubbleUpOperation(op, cmpIOp, indices, loc, rewriter);
  } else if (auto truncFOp = dyn_cast<arith::TruncFOp>(parentOp)) {
    bubbleUpOperation<arith::TruncFOp>(op, truncFOp, indices, loc, rewriter);
  } else if (auto extFOp = dyn_cast<arith::ExtFOp>(parentOp)) {
    bubbleUpOperation<arith::ExtFOp>(op, extFOp, indices, loc, rewriter);
  } else if (auto fpTosiOp = dyn_cast<arith::FPToSIOp>(parentOp)) {
    bubbleUpOperation<arith::FPToSIOp>(op, fpTosiOp, indices, loc, rewriter);
  } else if (auto siTofpOp = dyn_cast<arith::SIToFPOp>(parentOp)) {
    bubbleUpOperation<arith::SIToFPOp>(op, siTofpOp, indices, loc, rewriter);
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(parentOp)) {
    bubbleUpOperation<triton::ClampFOp>(op, clampFOp, indices, loc, rewriter);
  } else if (auto addFOp = dyn_cast<arith::AddFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::AddFOp>(op, addFOp, indices, loc, rewriter);
  } else if (auto subFOp = dyn_cast<arith::SubFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::SubFOp>(op, subFOp, indices, loc, rewriter);
  } else if (auto mulFOp = dyn_cast<arith::MulFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MulFOp>(op, mulFOp, indices, loc, rewriter);
  } else if (auto divFOp = dyn_cast<arith::DivFOp>(parentOp)) {
    bubbleUpFloatBinaryOp(op, divFOp, indices, loc, rewriter);
  } else if (auto minNumFOp = dyn_cast<arith::MinNumFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MinNumFOp>(op, minNumFOp, indices, loc,
                                            rewriter);
  } else if (auto maxNumFOp = dyn_cast<arith::MaxNumFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MaxNumFOp>(op, maxNumFOp, indices, loc,
                                            rewriter);
  } else if (auto cmpFOp = dyn_cast<arith::CmpFOp>(parentOp)) {
    bubbleUpOperation(op, cmpFOp, indices, loc, rewriter);
  } else if (auto broadCastOp = dyn_cast<triton::BroadcastOp>(parentOp)) {
    bubbleUpOperation(op, broadCastOp, indices, loc, rewriter);
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(parentOp)) {
    bubbleUpOperation<triton::ExpandDimsOp>(op, expandDimsOp, indices, loc,
                                            rewriter);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(parentOp)) {
    bubbleUpOperation<triton::SplatOp>(op, splatOp, indices, loc, rewriter);
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(parentOp)) {
    bubbleUpOperation<triton::MakeRangeOp>(op, makeRangeOp, indices, loc,
                                           rewriter);
  } else if (auto floorOp = dyn_cast<math::FloorOp>(parentOp)) {
    bubbleUpOperation<math::FloorOp>(op, floorOp, indices, loc, rewriter);
  } else if (auto ceilOp = dyn_cast<math::CeilOp>(parentOp)) {
    bubbleUpOperation<math::CeilOp>(op, ceilOp, indices, loc, rewriter);
  } else {
    return failure();
  }
  if (parentOp->use_empty())
    rewriter.eraseOp(parentOp);

  return success();
}

Value BubbleUpExtract::createExtractOp(Value value, ArrayRef<Value> indices,
                                       Location loc,
                                       PatternRewriter &rewriter) const {
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename BinOpTy>
void BubbleUpExtract::bubbleUpIntBinaryOp(Operation *op, BinOpTy binOp,
                                          ArrayRef<Value> indices, Location loc,
                                          PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(binOp.getLhs(), indices, loc, rewriter);
  auto rhs = createExtractOp(binOp.getRhs(), indices, loc, rewriter);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Binary\n" << *op << '\n' << binOp << '\n';
  });
  rewriter.replaceOpWithNewOp<BinOpTy>(op, lhs, rhs);
}

template <typename BinOpTy>
void BubbleUpExtract::bubbleUpFloatBinaryOp(Operation *op, BinOpTy binOp,
                                            ArrayRef<Value> indices,
                                            Location loc,
                                            PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(binOp.getLhs(), indices, loc, rewriter);
  auto rhs = createExtractOp(binOp.getRhs(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<BinOpTy>(op, lhs, rhs);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::ExtSIOp>(
    Operation *op, arith::ExtSIOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto in = createExtractOp(parentOp.getIn(), indices, loc, rewriter);
  auto resultType = cast<RankedTensorType>(parentOp.getOut().getType());
  rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resultType.getElementType(),
                                              in);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::CmpIOp>(
    Operation *op, arith::CmpIOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(parentOp.getLhs(), indices, loc, rewriter);
  auto rhs = createExtractOp(parentOp.getRhs(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, parentOp.getPredicateAttr(),
                                             lhs, rhs);
}

template <>
void BubbleUpExtract::bubbleUpOperation<triton::BroadcastOp>(
    Operation *op, triton::BroadcastOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
  SmallVector<Value> newIndices;
  for (const auto [index, shape] : llvm::zip_equal(indices, srcShape)) {
    if (shape == 1) {
      newIndices.push_back(
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
    } else {
      newIndices.push_back(index);
    }
  }
  auto extractedOp = createExtractOp(src, newIndices, loc, rewriter);
  rewriter.replaceOp(op, extractedOp);
}

template <>
void BubbleUpExtract::bubbleUpOperation<triton::ExpandDimsOp>(
    Operation *op, triton::ExpandDimsOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  SmallVector<Value> newIndices;
  for (const auto index : llvm::enumerate(indices)) {
    if (index.index() != parentOp.getAxis()) {
      newIndices.push_back(index.value());
    }
  }
  auto extractedOp = createExtractOp(src, newIndices, loc, rewriter);
  rewriter.replaceOp(op, extractedOp);
}

template <>
void BubbleUpExtract::bubbleUpOperation<triton::SplatOp>(
    Operation *op, triton::SplatOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  rewriter.replaceOp(op, src);
}

template <>
void BubbleUpExtract::bubbleUpOperation<triton::MakeRangeOp>(
    Operation *op, triton::MakeRangeOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto resultType = cast<RankedTensorType>(parentOp.getResult().getType());
  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      op, resultType.getElementType(), indices[0]);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::TruncFOp>(
    Operation *op, arith::TruncFOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto in = createExtractOp(parentOp.getIn(), indices, loc, rewriter);
  auto resultType = cast<RankedTensorType>(parentOp.getOut().getType());
  rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultType.getElementType(),
                                               in);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::ExtFOp>(
    Operation *op, arith::ExtFOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto in = createExtractOp(parentOp.getIn(), indices, loc, rewriter);
  auto resultType = cast<RankedTensorType>(parentOp.getOut().getType());
  rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, resultType.getElementType(),
                                               in);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::FPToSIOp>(
    Operation *op, arith::FPToSIOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto in = createExtractOp(parentOp.getIn(), indices, loc, rewriter);
  auto resultType = cast<RankedTensorType>(parentOp.getOut().getType());
  rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, resultType.getElementType(),
                                               in);
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::SIToFPOp>(
    Operation *op, arith::SIToFPOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto in = createExtractOp(parentOp.getIn(), indices, loc, rewriter);
  auto outType =
      cast<RankedTensorType>(parentOp.getOut().getType()).getElementType();
  rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, outType, in);
}

template <>
void BubbleUpExtract::bubbleUpOperation<triton::ClampFOp>(
    Operation *op, triton::ClampFOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto x = createExtractOp(parentOp.getX(), indices, loc, rewriter);
  auto min = createExtractOp(parentOp.getMin(), indices, loc, rewriter);
  auto max = createExtractOp(parentOp.getMax(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<triton::ClampFOp>(op, x, min, max,
                                                parentOp.getPropagateNan());
}

template <>
void BubbleUpExtract::bubbleUpOperation<arith::CmpFOp>(
    Operation *op, arith::CmpFOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(parentOp.getLhs(), indices, loc, rewriter);
  auto rhs = createExtractOp(parentOp.getRhs(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, parentOp.getPredicateAttr(),
                                             lhs, rhs);
}

template <>
void BubbleUpExtract::bubbleUpOperation<math::FloorOp>(
    Operation *op, math::FloorOp parentOp, ArrayRef<Value> indices,
    Location loc, PatternRewriter &rewriter) const {
  auto operand = createExtractOp(parentOp.getOperand(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<math::FloorOp>(op, operand,
                                             parentOp.getFastmath());
}

template <>
void BubbleUpExtract::bubbleUpOperation<math::CeilOp>(
    Operation *op, math::CeilOp parentOp, ArrayRef<Value> indices, Location loc,
    PatternRewriter &rewriter) const {
  auto operand = createExtractOp(parentOp.getOperand(), indices, loc, rewriter);
  rewriter.replaceOpWithNewOp<math::CeilOp>(op, operand,
                                            parentOp.getFastmath());
}

BubbleUpOperationPass::BubbleUpOperationPass(
    const BubbleUpOperationOptions &options)
    : BubbleUpOperationBase(options) {}

void BubbleUpOperationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<BubbleUpExtract>(ctx, enableAggressiveMode);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createBubbleUpOperationPass(const BubbleUpOperationOptions &options) {
  return std::make_unique<BubbleUpOperationPass>(options);
}