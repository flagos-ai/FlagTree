//===------------------- LinalgFusion.cpp --------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to fuse scalar input linalg operations for
// better performance. It applies scalar fusion transformations to reduce
// redundant memory read and write operations. Elementwise fusion needs to
// be implemented later.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/LinalgFusion/LinalgFusion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton-shared/Utils/Utils.h"
#include "utils/LinalgOpBuilderHelper.h"

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;

namespace {
template <typename ArithOp, typename MKOp>
struct BinaryScalarAndTensorOpFusion
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isValidUnaryElementWiseLinalgOp(linalg::LinalgOp op) const {
    if (!op.isSingleInputOutput() || !op.isAllParallelLoops() ||
        op.getNumLoops() < 1) {
      return false;
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::LinalgOp>(op);
    return regionOps.size() == 1 || isa<linalg::ReciprocalOp>(op);
  }

  // Check if the input can be traced back to a fill op through a chain of
  // unary elementwise linalg ops
  bool
  isScalarPropagationInput(Value input, Value &fillInput,
                           SmallVector<linalg::LinalgOp> &transformOps) const {
    // Check if the input is from a fill op or a chain of
    auto fillOp = input.getDefiningOp<linalg::FillOp>();
    if (fillOp) {
      fillInput = fillOp->getOperand(0);
      return true;
    }
    // Check if the input is from a valid unary elementwise linalg op
    auto linalgOp = input.getDefiningOp<linalg::LinalgOp>();
    if (linalgOp && isValidUnaryElementWiseLinalgOp(linalgOp)) {
      transformOps.push_back(linalgOp);
      return isScalarPropagationInput(linalgOp->getOperand(0), fillInput,
                                      transformOps);
    }
    return false;
  }

  // Create the scalar input by applying the chain of unary elementwise ops on
  // the fill input
  Value createScalarInput(Value input, PatternRewriter &rewriter,
                          Location loc) const {
    Value fillInput;
    SmallVector<linalg::LinalgOp> transformOps;
    if (!isScalarPropagationInput(input, fillInput, transformOps)) {
      return nullptr;
    }
    Value scalarInput = fillInput;
    for (int i = transformOps.size() - 1; i >= 0; i--) {
      auto op = transformOps[i];
      SmallVector<Value> arithInputs;
      auto regionOp = mlir::triton::getRegionOps<linalg::LinalgOp>(op).back();
      auto type = regionOp->getResultTypes()[0];
      if (isa<linalg::ReciprocalOp>(op))
        arithInputs.push_back(
            rewriter.create<arith::ConstantOp>(loc, rewriter.getOneAttr(type)));
      arithInputs.push_back(scalarInput);
      scalarInput = rewriter
                        .create(loc,
                                rewriter.getStringAttr(
                                    regionOp->getName().getStringRef()),
                                arithInputs, type)
                        ->getResult(0);
    }
    return scalarInput;
  }

  // Both inputs are constant propagation, create scalar operation and fill
  LogicalResult handleBothScalarInputs(linalg::GenericOp op,
                                       PatternRewriter &rewriter,
                                       Value scalarInput0,
                                       Value scalarInput1) const {
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    auto resultType = cast<RankedTensorType>(op.getResultTypes()[0]);
    auto loc = op->getLoc();
    auto scalarResult =
        rewriter
            .create(loc,
                    rewriter.getStringAttr(
                        regionOps.front()->getName().getStringRef()),
                    ValueRange{scalarInput0, scalarInput1},
                    resultType.getElementType())
            ->getResult(0);
    auto fillRes = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{scalarResult},
                                               op.getOutputs()[0])
                       ->getResult(0);
    rewriter.replaceAllOpUsesWith(op, fillRes);
    return success();
  }

  LogicalResult handleScalarAndTensorInput(linalg::GenericOp op,
                                           PatternRewriter &rewriter,
                                           Value scalarInput,
                                           Value tensorInput) const {
    auto resultType = cast<RankedTensorType>(op.getResultTypes()[0]);
    auto loc = op->getLoc();
    auto newRes =
        rewriter
            .create<MKOp>(op->getLoc(), op->getResultTypes()[0], tensorInput,
                          scalarInput, op.getOutputs()[0])
            .getResult(0);

    // Handle special case: subtraction operation with tensor input in the
    // second position.
    if (isa<arith::SubFOp>(op.getRegion().front().front()) &&
        tensorInput == op.getInputs()[1]) {
      newRes = buildLinalgElementwise<arith::NegFOp>(
          rewriter, op->getLoc(), resultType, ValueRange{newRes});
    }
    rewriter.replaceAllOpUsesWith(op, newRes);
    return success();
  }

  LogicalResult convertToIntegerDivision(linalg::GenericOp op,
                                         PatternRewriter &rewriter,
                                         Value dividend, Value divisor) const {
    auto divsi =
        rewriter.create<arith::DivSIOp>(op->getLoc(), dividend, divisor);
    auto siToFp = rewriter.create<arith::SIToFPOp>(
        op->getLoc(),
        cast<RankedTensorType>(op.getResultTypes()[0]).getElementType(), divsi);

    auto fillRes = rewriter
                       .create<linalg::FillOp>(op->getLoc(), ValueRange{siToFp},
                                               op.getOutputs()[0])
                       ->getResult(0);

    rewriter.replaceAllOpUsesWith(op, fillRes);
    return success();
  }

  // Linalg op DivF will convert to reciprocal + mul, scalar reciprocal
  // will convert to scalar divf which has low precision.
  LogicalResult handleDivisionCase(linalg::GenericOp op,
                                   PatternRewriter &rewriter,
                                   linalg::ReciprocalOp reciprocalOp) const {
    auto reciprocal = reciprocalOp->getResult(0);
    auto dividend =
        reciprocal == op.getInputs()[0] ? op.getInputs()[1] : op.getInputs()[0];

    Value dividendScalar;
    SmallVector<linalg::LinalgOp> dividendTransforms;
    // tensor + scalar/tensor recip: no conversion
    if (!isScalarPropagationInput(dividend, dividendScalar, dividendTransforms))
      return failure();

    Value recipScalar;
    SmallVector<linalg::LinalgOp> recipTransforms;
    // scalar + tensor recip -> mulvs
    if (!isScalarPropagationInput(reciprocalOp->getOperand(0), recipScalar,
                                  recipTransforms)) {
      return handleScalarAndTensorInput(op, rewriter, dividendScalar,
                                        reciprocal);
    }

    const bool isIntegerInput = isa<IntegerType>(dividendScalar.getType()) &&
                                isa<IntegerType>(recipScalar.getType());
    const bool hasSingleTransformOp =
        dividendTransforms.size() == 1 && recipTransforms.size() == 1;
    // Convert scalar + scalar divf + sitofp to divsi.
    if (isIntegerInput && hasSingleTransformOp &&
        isa<arith::SIToFPOp>(mlir::triton::getRegionOps<linalg::LinalgOp>(
                                 dividendTransforms.front())
                                 .front()) &&
        isa<arith::SIToFPOp>(mlir::triton::getRegionOps<linalg::LinalgOp>(
                                 recipTransforms.front())
                                 .front())) {
      return convertToIntegerDivision(op, rewriter, dividendScalar,
                                      recipScalar);
    }

    // scalar + scalar recip : no conversion
    return failure();
  }

  linalg::ReciprocalOp matchDivAndGetRecip(linalg::GenericOp op) const {
    auto lhsRecip = op->getOperand(0).getDefiningOp<linalg::ReciprocalOp>();
    auto rhsRecip = op->getOperand(1).getDefiningOp<linalg::ReciprocalOp>();
    assert(!(lhsRecip && rhsRecip) &&
           "Currently, we only handle cases where one input of the mul op is a "
           "reciprocal op.");
    if (auto reciprocalOp = lhsRecip ? lhsRecip : rhsRecip)
      return reciprocalOp;
    return nullptr;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isaElemwiseSingleBinaryOpInterface(op)) {
      return failure();
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    if (!isa<ArithOp>(regionOps.front())) {
      return failure();
    }

    assert(regionOps.size() == 1 &&
           "Expected a single operation in the linalg.generic region");

    // Handle multiplication special case with reciprocal:
    // DivF/DivSI convert to reciprocal + mul.
    if (std::is_same_v<ArithOp, arith::MulFOp>) {
      if (auto reciprocalOp = matchDivAndGetRecip(op)) {
        return handleDivisionCase(op, rewriter, reciprocalOp);
      }
    }

    auto input0 = op.getInputs()[0];
    auto input1 = op.getInputs()[1];
    auto loc = op->getLoc();
    // Try to create scalar input for both inputs
    Value scalarInput0 = createScalarInput(input0, rewriter, loc);
    Value scalarInput1 = createScalarInput(input1, rewriter, loc);
    if (!scalarInput1 && !scalarInput0) {
      return failure();
    }

    if (scalarInput0 && scalarInput1) {
      return handleBothScalarInputs(op, rewriter, scalarInput0, scalarInput1);
    }

    Value scalarInput = scalarInput1 ? scalarInput1 : scalarInput0;
    Value tensorInput = scalarInput1 ? input0 : input1;

    return handleScalarAndTensorInput(op, rewriter, scalarInput, tensorInput);
  }
};

struct LinalgFPToSIAndSIToFPCanonicalization
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isaElemwiseSingleUnaryOpInterface(op)) {
      return failure();
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    if (!isa<arith::SIToFPOp>(regionOps.front())) {
      return failure();
    }

    assert(regionOps.size() == 1 &&
           "Expected a single operation in the linalg.generic region");

    auto input = op.getInputs()[0];
    auto prevOp = input.getDefiningOp<linalg::GenericOp>();
    if (!prevOp) {
      return failure();
    }
    auto prevRegionOps = mlir::triton::getRegionOps<linalg::GenericOp>(prevOp);

    if (prevRegionOps.empty() || !isa<arith::FPToSIOp>(prevRegionOps.front())) {
      return failure();
    }

    if (op->getResultTypes()[0] == prevOp->getOperandTypes()[0]) {
      rewriter.replaceOp(op, prevOp->getOperand(0));
      return success();
    }

    if (dyn_cast<RankedTensorType>(op->getResultTypes()[0])
            .getElementTypeBitWidth() >
        dyn_cast<RankedTensorType>(prevOp->getOperandTypes()[0])
            .getElementTypeBitWidth()) {
      auto res = rewriter
                     .create<linalg::GenericOp>(
                         op->getLoc(), op.getResultTypes(),
                         prevOp->getOperand(0), op.getOutputs(),
                         op.getIndexingMapsArray(), op.getIteratorTypesArray(),
                         [](OpBuilder &b, Location loc, ValueRange args) {
                           auto extf = b.create<arith::ExtFOp>(
                               loc, args.back().getType(), args.front());
                           b.create<linalg::YieldOp>(loc, extf.getResult());
                         })
                     .getResult(0);
      rewriter.replaceOp(op, res);
      return success();
    }

    auto res = rewriter
                   .create<linalg::GenericOp>(
                       op->getLoc(), op.getResultTypes(), prevOp->getOperand(0),
                       op.getOutputs(), op.getIndexingMapsArray(),
                       op.getIteratorTypesArray(),
                       [](OpBuilder &b, Location loc, ValueRange args) {
                         auto truncf = b.create<arith::TruncFOp>(
                             loc, args.back().getType(), args.front());
                         b.create<linalg::YieldOp>(loc, truncf.getResult());
                       })
                   .getResult(0);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct LinalgExtSIAndSIToFPFusion : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isaElemwiseSingleUnaryOpInterface(op)) {
      return failure();
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    if (!isa<arith::SIToFPOp>(regionOps.front())) {
      return failure();
    }

    assert(regionOps.size() == 1 &&
           "Expected a single operation in the linalg.generic region");

    auto input = op.getInputs()[0];
    auto prevOp = input.getDefiningOp<linalg::GenericOp>();
    if (!prevOp) {
      return failure();
    }
    auto prevRegionOps = mlir::triton::getRegionOps<linalg::GenericOp>(prevOp);

    if (prevRegionOps.empty() || !isa<arith::ExtSIOp>(prevRegionOps.front())) {
      return failure();
    }

    auto newOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(), op.getResultTypes(), prevOp->getOperand(0),
        op.getOutputs()[0], op.getIndexingMapsArray(),
        op.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          auto siToFp = b.create<arith::SIToFPOp>(loc, args.back().getType(),
                                                  args.front());
          b.create<linalg::YieldOp>(loc, siToFp.getResult());
        });
    rewriter.replaceOp(op, newOp.getResult(0));
    return success();
  }
};

template <typename TensorShapeOp>
struct TensorShapeOpAndLinalgSIToFPCanonicalization
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isaElemwiseSingleUnaryOpInterface(op)) {
      return failure();
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    if (!isa<arith::SIToFPOp>(regionOps.front())) {
      return failure();
    }

    assert(regionOps.size() == 1 &&
           "Expected a single operation in the linalg.generic region");

    auto input = op.getInputs()[0];
    auto prevOp = input.getDefiningOp<TensorShapeOp>();
    if (!prevOp) {
      return failure();
    }

    auto outputType =
        cast<RankedTensorType>(op->getResultTypes()[0]).getElementType();
    auto outputShape =
        cast<RankedTensorType>(prevOp->getOperandTypes()[0]).getShape();
    auto outputTensor =
        rewriter.create<tensor::EmptyOp>(op->getLoc(), outputShape, outputType);

    auto rank = outputShape.size();
    SmallVector<AffineMap, 2> indexingMaps(
        2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    auto sitofp = rewriter
                      .create<linalg::GenericOp>(
                          op->getLoc(), TypeRange{outputTensor.getType()},
                          ValueRange{prevOp->getOperand(0)},
                          ValueRange{outputTensor}, indexingMaps, iterators,
                          [](OpBuilder &b, Location loc, ValueRange args) {
                            auto siToFp = b.create<arith::SIToFPOp>(
                                loc, args.back().getType(), args.front());
                            b.create<linalg::YieldOp>(loc, siToFp.getResult());
                          })
                      ->getResults()[0];
    auto newShape = rewriter.create<TensorShapeOp>(
        op->getLoc(), op.getResultTypes()[0], sitofp,
        prevOp.getReassociationIndices());
    rewriter.replaceOp(op, newShape.getResult());
    return success();
  }
};

struct LinalgFillSIToFPFusion : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isaElemwiseSingleUnaryOpInterface(op)) {
      return failure();
    }
    auto regionOps = mlir::triton::getRegionOps<linalg::GenericOp>(op);
    if (!isa<arith::SIToFPOp>(regionOps.front())) {
      return failure();
    }

    assert(regionOps.size() == 1 &&
           "Expected a single operation in the linalg.generic region");

    auto input = op.getInputs()[0];
    auto fillOp = input.getDefiningOp<linalg::FillOp>();
    if (!fillOp) {
      return failure();
    }

    auto outputType =
        cast<RankedTensorType>(op->getResultTypes()[0]).getElementType();
    Value fillScalar = fillOp.getInputs()[0];
    Value siToFpScalar =
        rewriter.create<arith::SIToFPOp>(op.getLoc(), outputType, fillScalar);
    auto newFill =
        rewriter
            .create<linalg::FillOp>(op.getLoc(), ValueRange{siToFpScalar},
                                    op.getOutputs()[0])
            ->getResult(0);

    rewriter.replaceOp(op, newFill);
    return success();
  }
};

struct ArithExtSISitofpFusion : public OpRewritePattern<arith::SIToFPOp> {
  using OpRewritePattern<arith::SIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto extOp = op.getOperand().getDefiningOp<arith::ExtSIOp>();
    if (!extOp)
      return failure();

    auto newSitofp = rewriter.create<arith::SIToFPOp>(op.getLoc(), op.getType(),
                                                      extOp.getOperand());
    rewriter.replaceOp(op, newSitofp.getResult());
    return success();
  }
};
} // namespace

void mlir::triton::populateLinalgBinaryOpFusionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BinaryScalarAndTensorOpFusion<arith::AddFOp, mk::AddVS>>(
      patterns.getContext());
  patterns.add<BinaryScalarAndTensorOpFusion<arith::SubFOp, mk::SubVS>>(
      patterns.getContext());
  patterns.add<BinaryScalarAndTensorOpFusion<arith::MulFOp, mk::MulVS>>(
      patterns.getContext());
}

void mlir::triton::populateLinalgTypeConversionFusionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgFPToSIAndSIToFPCanonicalization>(patterns.getContext());
  patterns.add<LinalgExtSIAndSIToFPFusion>(patterns.getContext());
  patterns.add<
      TensorShapeOpAndLinalgSIToFPCanonicalization<tensor::ExpandShapeOp>,
      TensorShapeOpAndLinalgSIToFPCanonicalization<tensor::CollapseShapeOp>>(
      patterns.getContext());
  patterns.add<LinalgFillSIToFPFusion>(patterns.getContext());
  patterns.add<ArithExtSISitofpFusion>(patterns.getContext());
}

// TODO: Support linalg elementwise op fusion.
#if 0
void mlir::triton::populateLinalgFusionPatterns(RewritePatternSet &patterns) {
  // Add folding with reshape by expansion patterns.
  linalg::ControlFusionFn defaultControlFn = [](OpOperand *fusedOperand) {
    return false;
  };
  linalg::populateElementwiseOpsFusionPatterns(patterns, defaultControlFn);
}
#endif
