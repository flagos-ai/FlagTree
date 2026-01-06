//===------------------- LinalgToMK.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "utils/FusionHelper.h"
#include "utils/LinalgOpBuilderHelper.h"
#include "utils/ReduceScanCommon.h"
#include "utils/utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "linalg-to-mk"

using namespace mlir;
using namespace mk;

#define GEN_PASS_CLASSES
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

namespace {

bool isConstantTensor(Value &v, double targetValue, bool isApprox) {
  auto fillOp = dyn_cast<linalg::FillOp>(v.getDefiningOp());
  if (!fillOp) {
    return false;
  }

  auto fillValue = fillOp.getInputs()[0];
  auto constOp = fillValue.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
    return isApprox ? (std::abs(val.getValueAsDouble() - targetValue) < 1e-5)
                    : (val.getValueAsDouble() == targetValue);
  }
  if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
    return val.getValue() == static_cast<int64_t>(targetValue);
  }

  return false;
}

// Check if the given value is a tensor filled with 0.
bool isZeroTensor(Value &v) { return isConstantTensor(v, 0.0); }

// Check if the given value is a tensor filled with 1.
bool isOneTensor(Value &v) { return isConstantTensor(v, 1.0); }

bool isHalfTensor(Value &v) {
  const float halfValue = 0.5f;
  return isConstantTensor(v, halfValue);
}

bool isTwoTensor(Value &v) {
  // Check if the value is a constant tensor with the value of 2.0.
  const float twoValue = 2.0f;
  return isConstantTensor(v, twoValue);
};

bool checkReductionBaseAttr(linalg::ReduceOp op, OpBuilder &builder,
                            TypedAttr &attr) {
  auto out = op.getInits().front();

  auto outDef = out.getDefiningOp();
  if (!outDef) {
    return false;
  }
  Value value;
  if (isa<linalg::FillOp>(outDef)) {
    value = cast<linalg::FillOp>(outDef).value();
  } else if (isa<tensor::InsertOp>(outDef)) {
    value = cast<tensor::InsertOp>(outDef).getScalar();
  } else {
    // If the output is not a fill or insert op, we cannot determine the
    // reduction base attribute.
    return false;
  }
  auto valueDef = value.getDefiningOp<arith::ConstantOp>();
  if (!valueDef) {
    return false;
  }

  return valueDef.getValueAttr() == attr;
}

static linalg::ReduceOp createReduceOp(OpBuilder &rewriter, linalg::ReduceOp op,
                                       Location loc, ValueRange sources,
                                       SmallVector<int64_t> dims,
                                       SmallVector<int64_t> shape,
                                       Type elementType, TypedAttr attr) {
  Value init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);

  auto accBaseConstOp =
      rewriter.create<arith::ConstantOp>(loc, elementType, attr);
  Value initTensor = rewriter
                         .create<linalg::FillOp>(
                             loc, ValueRange{accBaseConstOp}, ValueRange{init})
                         .result();

  return rewriter.create<linalg::ReduceOp>(
      loc, sources, ValueRange{initTensor}, dims,
      [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
        assert(inputs.size() == 2);

        auto reduceBlock = op.getBody();
        IRMapping mapping;
        mapping.map(reduceBlock->getArguments(), inputs);

        for (auto &op : reduceBlock->without_terminator()) {
          opBuilder.clone(op, mapping);
        }

        auto yield = reduceBlock->getTerminator();
        auto results =
            llvm::map_to_vector(yield->getOperands(),
                                [&](Value val) { return mapping.lookup(val); });

        opBuilder.create<linalg::YieldOp>(loc, results);
      });
};

// Convert linalg.matmul to mk.dot
struct LinalgMatmulOpRewrite : public OpRewritePattern<linalg::MatmulOp> {
private:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  Value channelNorm(PatternRewriter &rewriter, Location loc, Value src) const {
    auto type = cast<RankedTensorType>(src.getType());
    auto elementType = type.getElementType();
    int64_t alignBase = elementType.getIntOrFloatBitWidth() == 8 ? 128 : 64;
    int64_t M = type.getShape()[0];
    int64_t N = type.getShape()[1];

    assert(N >= 4 && llvm::isPowerOf2_64(N) &&
           "N must be at least 4 and a power of 2");

    int64_t N1 = std::max(1L, N / alignBase);
    int64_t N2 = std::min(alignBase, N);

    if (N1 == 1)
      return rewriter.create<tensor::ExpandShapeOp>(
          loc, type.clone({1, M, N}), src,
          ArrayRef<ReassociationIndices>{{0, 1}, {2}});

    Value reshpe = rewriter.create<tensor::ExpandShapeOp>(
        loc, type.clone({M, N1, N2}), src,
        ArrayRef<ReassociationIndices>{{0}, {1, 2}});
    auto permsTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{N1, M, N2}, elementType);
    return rewriter
        .create<linalg::TransposeOp>(loc, reshpe, permsTensor,
                                     ArrayRef<int64_t>{1, 0, 2})
        ->getResult(0);
  }

  Value dechannelNorm(PatternRewriter &rewriter, Location loc,
                      Value src) const {
    auto type = cast<RankedTensorType>(src.getType());
    auto elementType = type.getElementType();
    int64_t N1 = type.getShape()[0];
    int64_t M = type.getShape()[1];
    int64_t N2 = type.getShape()[2];

    if (N1 == 1)
      return rewriter.create<tensor::CollapseShapeOp>(
          loc, src, ArrayRef<ReassociationIndices>{{0, 1}, {2}});

    auto permsTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{M, N1, N2}, elementType);
    Value transpose = rewriter
                          .create<linalg::TransposeOp>(
                              loc, src, permsTensor, ArrayRef<int64_t>{1, 0, 2})
                          ->getResult(0);
    return rewriter.create<tensor::CollapseShapeOp>(
        loc, transpose, ArrayRef<ReassociationIndices>{{0}, {1, 2}});
  }

  // Find the `scf.for` loop in which the current `linalg.matmul` is used as the
  // iteration variable.
  std::optional<std::pair<scf::ForOp, unsigned>>
  findForOpAndIdx(linalg::MatmulOp op) const {
    if (!op.getResult(0).hasOneUse())
      return std::nullopt;

    auto U = op.getResult(0).use_begin();
    auto idx = U->getOperandNumber();
    auto yieldOp = dyn_cast<scf::YieldOp>(U->getOwner());

    if (!yieldOp)
      return std::nullopt;

    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());

    if (!forOp)
      return std::nullopt;

    auto iterArg = forOp.getRegionIterArgs()[idx];

    if (!iterArg.hasOneUse() || op.getOutputs()[0] != iterArg)
      return std::nullopt;

    return std::make_pair(forOp, idx);
  }

public:
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto a = channelNorm(rewriter, loc, op.getInputs()[0]);
    auto b = channelNorm(rewriter, loc, op.getInputs()[1]);
    auto fillOp = op.getOutputs()[0].getDefiningOp<linalg::FillOp>();

    if (fillOp && matchPattern(fillOp.getInputs()[0], m_AnyZeroFloat())) {
      Value res = rewriter.create<tensor::EmptyOp>(
          loc, channelNorm(rewriter, loc, op.getOutputs()[0]).getType(),
          ValueRange{});
      res = rewriter
                .create<mk::DotOp>(loc, res.getType(), a, b, res,
                                   false /* en_psum */)
                ->getResult(0);
      rewriter.replaceOp(op, dechannelNorm(rewriter, loc, res));
    } else if (auto optFromLoopInfo = findForOpAndIdx(op)) {
      // If we find that the current `linalg.matmul` is the loop's iteration
      // variable, we can hoist the accumulation matrix "channelNorm" outside
      // the loop and sink the "dechannelNorm" outside as well.
      auto [forOp, idx] = *optFromLoopInfo;
      SmallVector<Value> inits = forOp.getInits();
      rewriter.setInsertionPoint(forOp);
      inits[idx] = channelNorm(rewriter, loc, inits[idx]);
      auto newForOp = rewriter.create<scf::ForOp>(
          forOp->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), inits);
      auto body = newForOp.getBody();
      body->getOperations().splice(body->begin(),
                                   forOp.getBody()->getOperations());
      forOp.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        if (i == idx)
          continue;
        forOp.getRegionIterArg(i).replaceAllUsesWith(
            newForOp.getRegionIterArg(i));
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      }

      rewriter.setInsertionPoint(op);
      auto dot = rewriter.create<mk::DotOp>(
          loc, newForOp->getResultTypes()[idx], a, b,
          newForOp.getRegionIterArg(idx), true /* en_psum */);
      body->getTerminator()->setOperand(idx, dot->getResult(0));
      rewriter.eraseOp(op);

      rewriter.setInsertionPointAfter(newForOp);
      Value res =
          dechannelNorm(rewriter, forOp->getLoc(), newForOp->getResult(idx));
      forOp->getResult(idx).replaceAllUsesWith(res);
      rewriter.eraseOp(forOp);
    } else {
      Value output = channelNorm(rewriter, loc, op.getOutputs()[0]);
      output = rewriter
                   .create<mk::DotOp>(loc, output.getType(), a, b, output,
                                      true /* en_psum */)
                   ->getResult(0);
      rewriter.replaceOp(op, dechannelNorm(rewriter, loc, output));
    }

    return success();
  }
};

struct MKDotScaleOpRewrite : public OpRewritePattern<mk::DotScaledOp> {
private:
  using OpRewritePattern<mk::DotScaledOp>::OpRewritePattern;

  Value buildExtF(OpBuilder &rewriter, Location loc, Value input,
                  Type targetType) const {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto empty =
        rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(), targetType);

    auto rank = inputType.getRank();

    auto identityMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps(2, identityMap);
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    return rewriter
        .create<linalg::GenericOp>(
            loc, TypeRange{empty.getType()}, ValueRange{input},
            ValueRange{empty}, indexingMaps, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedloc,
                ValueRange iterArgs) {
              auto extf = nestedBuilder.create<arith::ExtFOp>(
                  nestedloc, targetType, iterArgs[0]);
              nestedBuilder.create<linalg::YieldOp>(nestedloc,
                                                    ValueRange{extf});
            })
        ->getResult(0);
  }

  Type dotTypeFromAttr(triton::ScaleDotElemType type, MLIRContext *ctx) const {
    switch (type) {
    case triton::ScaleDotElemType::E4M3:
      return Float8E4M3FNType::get(ctx);
    case triton::ScaleDotElemType::E5M2:
      return Float8E5M2Type::get(ctx);
    case triton::ScaleDotElemType::E2M1:
      return Float4E2M1FNType::get(ctx);
    case triton::ScaleDotElemType::BF16:
      return BFloat16Type::get(ctx);
    case triton::ScaleDotElemType::FP16:
      return Float16Type::get(ctx);
    // unsupported
    // case triton::ScaleDotElemType::E2M3:
    //   return Float6E2M3FNType::get(ctx);
    // case triton::ScaleDotElemType::E3M2:
    //   return Float6E3M2FNType::get(ctx);
    default:
      llvm_unreachable("unsupported type!");
    }
  }

  Value upcast(OpBuilder &rewriter, Location loc, Value v, Value vScale,
               Type vElemType, Type compType, bool transposed) const {

    if (vElemType == compType)
      return v;

    if (!vScale)
      return buildExtF(rewriter, loc, v, compType);

    Value scaleInput = v;
    auto tensorType = cast<RankedTensorType>(v.getType());
    // Since b matrix scale is stored in column major, need transpose
    if (transposed) {

      auto originShape = tensorType.getShape();
      SmallVector<int64_t> perm{1, 0};
      SmallVector<int64_t> transposeShape = {originShape[1], originShape[0]};
      auto empty = rewriter.create<tensor::EmptyOp>(
          loc, transposeShape, tensorType.getElementType());
      scaleInput = rewriter.create<linalg::TransposeOp>(loc, v, empty, perm)
                       ->getResult(0);
    }

    // exsample:
    // tt.dot_scaled %26, %22 scale %25, %cst_0 lhs = e4m3 rhs = e2m1
    // : tensor<4x32xf8E4M3FN> * tensor<16x4xi8>, tensor<4x1xi8> ->
    // tensor<4x4xf32>
    if (tensorType.getElementType().isInteger(8) &&
        isa<Float4E2M1FNType>(vElemType)) {
      auto shape = cast<RankedTensorType>(scaleInput.getType()).getShape();
      SmallVector<int64_t> newShape(shape);
      newShape.back() *= 2;
      scaleInput = rewriter.create<mk::BitcastOp>(
          loc,
          RankedTensorType::get(newShape,
                                Float4E2M1FNType::get(vElemType.getContext())),
          scaleInput);
    }

    assert(cast<RankedTensorType>(scaleInput.getType()).getElementType() ==
           vElemType);

    scaleInput = buildExtF(rewriter, loc, scaleInput, compType);

    // dstBuffer inplace
    if (vScale) {
      auto shape = cast<RankedTensorType>(scaleInput.getType()).getShape();
      auto empty = rewriter.create<tensor::EmptyOp>(loc, shape, compType);
      scaleInput = rewriter
                       .create<mk::DequantOp>(
                           loc, RankedTensorType::get(shape, compType),
                           scaleInput, vScale, empty)
                       ->getResult(0);
    }

    // read: dstBuffer, write: transposeMid
    if (transposed) {
      // e2m1 need use shape after bitcast
      auto shape = cast<RankedTensorType>(scaleInput.getType()).getShape();
      assert(shape.size() == 2);
      SmallVector<int64_t> newShape{shape[1], shape[0]};
      auto empty = rewriter.create<tensor::EmptyOp>(loc, newShape, compType);
      SmallVector<int64_t> perm{1, 0};
      scaleInput =
          rewriter.create<linalg::TransposeOp>(loc, scaleInput, empty, perm)
              ->getResult(0);
    }

    return scaleInput;
  }

public:
  LogicalResult matchAndRewrite(mk::DotScaledOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value dst = op.getDst();
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    auto aElemAttr = op.getAElemType();
    auto bElemAttr = op.getBElemType();

    // Cast input to compType
    auto aElemType = dotTypeFromAttr(aElemAttr, rewriter.getContext());
    auto bElemType = dotTypeFromAttr(bElemAttr, rewriter.getContext());

    assert(cast<RankedTensorType>(a.getType()).getRank() == 2 &&
           "support rank is 2 only");

    Type compType = (aElemType.isF16() || bElemType.isF16())
                        ? rewriter.getF16Type()
                        : rewriter.getBF16Type();

    // If has scale, do quantization

    a = upcast(rewriter, loc, a, aScale, aElemType, compType, false);
    b = upcast(rewriter, loc, b, bScale, bElemType, compType, true);

    // Do standard matmul
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, op->getResultTypes(), ValueRange{a, b}, ValueRange{dst});

    rewriter.replaceOp(op, matmulOp->getResults());

    return success();
  }
};

struct NormalizeReduceInitToIdentityPattern
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  Operation *accumulateInit(OpBuilder &builder, linalg::ReduceOp op,
                            Value reduceVal, Location loc) const {
    Value init = op.getInits()[0];
    auto outputType = cast<RankedTensorType>(init.getType());
    auto rank = outputType.getRank();
    SmallVector<AffineMap> idMaps(3, builder.getMultiDimIdentityMap(rank));
    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    auto genericOp = builder.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{reduceVal, init},
        ValueRange{init}, idMaps, iterators);
    genericOp.getRegion().takeBody(op.getRegion());
    genericOp.getRegion().front().addArgument(outputType.getElementType(), loc);

    return genericOp;
  }

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {

    auto reduceOps = getRegionOps<linalg::ReduceOp>(op);

    auto *reduceOp = reduceOps.front();
    if (reduceOps.size() != 1)
      return failure();

    // If the init value is reduction op base(reduction operation identity
    // value), don't need to accumulate it
    auto resType = cast<RankedTensorType>(op.getInits()[0].getType());
    auto constantType = resType.getElementType();
    auto inputType = cast<RankedTensorType>(op.getInputs()[0].getType());

    // TODO: Config according backend
    // NOTE: Assume has done integer to float
    if (!(isReductionOpAndTypeSupportedByTarget(reduceOp, constantType) ||
          isReduceToElementWiseOpAndTypeSupportedByTarget(
              reduceOp, constantType, inputType.getNumElements(),
              inputType.getRank()))) {
      return failure();
    }

    auto attr = getRedBaseAttr(rewriter, reduceOp, constantType);
    if (checkReductionBaseAttr(op, rewriter, attr))
      return failure();

    auto loc = op.getLoc();
    SmallVector<int64_t> dims(op.getDimensions().begin(),
                              op.getDimensions().end());
    SmallVector<int64_t> shape(resType.getShape().begin(),
                               resType.getShape().end());
    Value finalResult = createReduceOp(rewriter, op, loc, op.getInputs(), dims,
                                       shape, constantType, attr)
                            ->getResult(0);

    auto newOp = accumulateInit(rewriter, op, finalResult, loc);

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// todo : Determine whether precision promotion is needed based on hardware
// characteristics, eg, fp16->fp32.
// Tsingmicro does not require precision promotion since it is handled in
// hardware computation.
struct ReducePrecisionPromotionRewrite
    : public OpRewritePattern<linalg::ReduceOp> {

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() <
               cast<FloatType>(Float32Type::get(elemType.getContext()))
                   .getWidth() &&
           (isa<arith::AddFOp>(redOp) || isa<arith::MulFOp>(redOp));
  }
};

struct LinalgReduceToMKReduceConversion
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  SmallVector<int64_t, 4> reshapeReduceShapeTo4d(ArrayRef<int64_t> inputShape,
                                                 int64_t dim) const {

    auto rank = inputShape.size();
    assert(dim < rank && "Dim out of range");

    SmallVector<int64_t, 4> newShape;
    int64_t leftDimsElement = 1;
    int64_t rightDimsElement = 1;

    for (int i = 0; i < dim; i++)
      leftDimsElement *= inputShape[i];

    if (dim == inputShape.size() - 1)
      return {1, 1, leftDimsElement, inputShape[dim]};

    for (int i = dim + 1; i < rank; i++)
      rightDimsElement *= inputShape[i];

    newShape = {1, leftDimsElement, inputShape[dim], rightDimsElement}; // NHWC
    return newShape;
  }

  Value createReshapeOp(OpBuilder &rewriter, Location loc, Value src,
                        ArrayRef<int64_t> outputShape, Type elementType) const {

    int32_t rank = outputShape.size();

    auto shapeTensorShape = SmallVector<int64_t>{rank};
    Value shapeTensor = rewriter.create<tensor::EmptyOp>(loc, shapeTensorShape,
                                                         rewriter.getI32Type());

    for (int32_t i = 0; i < rank; i++) {
      auto index = rewriter.create<arith::ConstantIndexOp>(loc, i);
      auto dim = rewriter.create<arith::ConstantIntOp>(loc, outputShape[i],
                                                       rewriter.getI32Type());
      shapeTensor = rewriter.create<tensor::InsertOp>(loc, dim, shapeTensor,
                                                      ValueRange{index});
    }
    return rewriter.create<tensor::ReshapeOp>(
        loc, RankedTensorType::get(outputShape, elementType), src, shapeTensor);
  }

  Value convertToMKReduce(OpBuilder &rewriter, Location loc,
                          Operation *reduceOp, Value channelNormedInput,
                          SmallVector<int64_t> &inputShape4D, Type elementType,
                          bool lastDimReduce) const {

    auto newReduceDim = lastDimReduce ? 3 : 2;

    auto channelNormedShape =
        cast<RankedTensorType>(channelNormedInput.getType()).getShape();

    // Reduce output
    SmallVector<int64_t> channelNormedReduceOutput4D =
        lastDimReduce ? SmallVector<int64_t>{inputShape4D[0], inputShape4D[1],
                                             inputShape4D[2], 4}
                      : SmallVector<int64_t>{1, channelNormedShape[0],
                                             channelNormedShape[1],
                                             channelNormedShape[3]};

    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, channelNormedReduceOutput4D, elementType);
    auto outputType =
        RankedTensorType::get(channelNormedReduceOutput4D, elementType);
    ArrayAttr nhwcShape = rewriter.getI64ArrayAttr(inputShape4D);

    if (isa<arith::AddFOp>(reduceOp)) {
      return rewriter
          .create<mk::ReduceSumOp>(loc, outputType, channelNormedInput, empty,
                                   nhwcShape, newReduceDim)
          ->getResult(0);
    }
    if (isa<arith::MaximumFOp, arith::MaxNumFOp>(reduceOp)) {
      return rewriter
          .create<mk::ReduceMaxOp>(loc, outputType, channelNormedInput, empty,
                                   nhwcShape, newReduceDim)
          ->getResult(0);
    }
    if (isa<arith::MinimumFOp, arith::MinNumFOp>(reduceOp)) {
      return rewriter
          .create<mk::ReduceMinOp>(loc, outputType, channelNormedInput, empty,
                                   nhwcShape, newReduceDim)
          ->getResult(0);
    }
    llvm_unreachable("Unsupported reduction operation");
    return nullptr;
  }

  Value channelNorm(PatternRewriter &rewriter, Location loc, Value src,
                    SmallVector<int64_t> &inputShape4D) const {

    auto type = cast<RankedTensorType>(src.getType());
    auto elementType = type.getElementType();

    int64_t alignBase = elementType.getIntOrFloatBitWidth() == 8 ? 128 : 64;

    int lastDim = inputShape4D.back();
    // Triton always assume shape is power of 2
    assert(llvm::isPowerOf2_64(lastDim) && "LastDim must be power of 2");

    if (lastDim > alignBase) {

      // {1, H, W, C} -> {H, W, C}
      SmallVector<int64_t, 3> collapseShape(inputShape4D.begin() + 1,
                                            inputShape4D.end());
      Value collapse = rewriter.create<tensor::CollapseShapeOp>(
          loc, type.clone(collapseShape), src,
          ArrayRef<ReassociationIndices>{{0, 1}, {2}, {3}});

      // {H, W, C} -> {H, W, CX, C}
      int64_t cx = lastDim / alignBase;
      SmallVector<int64_t, 4> expandShape{collapseShape[0], collapseShape[1],
                                          cx, alignBase};

      Value expand = rewriter.create<tensor::ExpandShapeOp>(
          loc, type.clone(expandShape), collapse,
          ArrayRef<ReassociationIndices>{{0}, {1}, {2, 3}});

      // {H, W, CX, C} -> {CX, H, W, C}
      SmallVector<int64_t, 4> channelnormShape = {
          expandShape[2], expandShape[0], expandShape[1], expandShape[3]};
      auto permutateTensor =
          rewriter.create<tensor::EmptyOp>(loc, channelnormShape, elementType);
      auto channelNorm =
          rewriter
              .create<linalg::TransposeOp>(loc, expand, permutateTensor,
                                           ArrayRef<int64_t>{2, 0, 1, 3})
              ->getResult(0);

      return channelNorm;
    } else if (lastDim < 4) {
      // {1, H, W, C} -> {1, H, W, 4}
      SmallVector<int64_t> channelnormShape = inputShape4D;
      channelnormShape.back() = 4;

      auto empty =
          rewriter.create<tensor::EmptyOp>(loc, channelnormShape, elementType);

      auto insert = rewriter.create<tensor::InsertSliceOp>(
          loc, RankedTensorType::get(channelnormShape, elementType), src, empty,
          ValueRange(), ValueRange(), ValueRange(),
          ArrayRef<int64_t>{0, 0, 0, 0}, inputShape4D,
          ArrayRef<int64_t>{1, 1, 1, 1});
      return insert;
    }

    return src;
  }

  Value dechannelNorm(PatternRewriter &rewriter, Location loc, Value src,
                      SmallVector<int64_t> &inputShape4D,
                      bool lastDimReduce) const {

    auto type = cast<RankedTensorType>(src.getType());
    auto elementType = type.getElementType();
    int64_t alignBase = elementType.getIntOrFloatBitWidth() == 8 ? 128 : 64;

    int lastDim = inputShape4D.back();

    SmallVector<int64_t> outputShape =
        lastDimReduce
            ? SmallVector<int64_t>{1, 1, inputShape4D[2], 1}
            : SmallVector<int64_t>{1, 1, inputShape4D[1], inputShape4D.back()};

    if (4 <= lastDim && lastDim <= 128 && !lastDimReduce)
      return src;

    if (!lastDimReduce && lastDim > alignBase) {
      // {1, cx, left, c0} -> {1, left cx, c0}

      SmallVector<int64_t> permutationShape = {1, inputShape4D[1],
                                               lastDim / alignBase, alignBase};
      auto permutateTensor =
          rewriter.create<tensor::EmptyOp>(loc, permutationShape, elementType);
      return rewriter
          .create<linalg::TransposeOp>(loc, src, permutateTensor,
                                       ArrayRef<int64_t>{0, 2, 1, 3})
          ->getResult(0);
    }

    return rewriter.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get(outputShape, elementType), src, ValueRange(),
        ValueRange(), ValueRange(), ArrayRef<int64_t>{0, 0, 0, 0}, outputShape,
        ArrayRef<int64_t>{1, 1, 1, 1});
  }

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {

    auto reduceOps = getRegionOps<linalg::ReduceOp>(op);

    auto *reduceOp = reduceOps.front();
    if (reduceOps.size() != 1)
      return failure();
    // TODO: Config according backend
    // Assume has done integer to float conversion
    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto elementType = inputType.getElementType();
    if (!isReductionOpAndTypeSupportedByTarget(reduceOp, elementType)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported reduction operation or type.");
    }

    auto dims = op.getDimensions();
    if (dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support one dim reduce.");

    auto dim = dims[0];
    auto loc = op->getLoc();

    // Don't do channel norm for output since we always canonicalize init to
    // identity value
    auto attr = getRedBaseAttr(rewriter, reduceOp, elementType);
    // WORKAROUND: si to fp will generate uninitialized init value, also thought
    // as base identity value
    if (!(checkReductionBaseAttr(op, rewriter, attr) ||
          op.getInits().back().getDefiningOp<tensor::EmptyOp>()))
      return rewriter.notifyMatchFailure(
          op, "Init is not reduction op base value.");

    auto inputShape = inputType.getShape();
    bool lastDimReduce = (dim == inputShape.size() - 1);
    // Reshape input to 4D
    SmallVector<int64_t> inputShape4D = reshapeReduceShapeTo4d(inputShape, dim);
    Value input4D = createReshapeOp(rewriter, loc, op.getInputs()[0],
                                    inputShape4D, elementType);
    Value channelNormedInput =
        channelNorm(rewriter, loc, input4D, inputShape4D);

    Value reduce =
        convertToMKReduce(rewriter, loc, reduceOp, channelNormedInput,
                          inputShape4D, elementType, lastDimReduce);

    auto dechannelNormOutput =
        dechannelNorm(rewriter, loc, reduce, inputShape4D, lastDimReduce);

    auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);
    if (resultType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, dechannelNormOutput,
          ArrayRef<ReassociationIndices>{});
      return success();
    }

    Value result = createReshapeOp(rewriter, loc, dechannelNormOutput,
                                   resultType.getShape(), elementType);

    rewriter.replaceOp(op, result);
    // Implement layout transformation for ReduceOp here
    return success();
  }
};

template <typename ScalarGlobalLoadOp>
struct ScalarGlobalLoadRewrite : public OpRewritePattern<ScalarGlobalLoadOp> {
  using OpRewritePattern<ScalarGlobalLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScalarGlobalLoadOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement memory space check
    if (triton::isOperandMemorySpaceSPM(op.getMemref()))
      return rewriter.notifyMatchFailure(op, "Not global memory load");

    auto indices = op.getIndices();
    if (indices.size() > 1) {
      return rewriter.notifyMatchFailure(op, "Load has multiple indices");
    }

    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto index = indices.empty() ? zero : indices[0];

    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);

    auto loc = op.getLoc();
    auto type = cast<MemRefType>(op.getMemref().getType());
    auto src = rewriter.create<memref::SubViewOp>(
        loc, op.getMemref(), ValueRange{index}, ValueRange(one),
        ValueRange(one));

    auto tensorType = RankedTensorType::get({1}, type.getElementType());
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, src, true /* restrict */, true /* writable */);
    auto empty = rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
    // FIXME: Bufferization pass insert copy op according address space?
    auto copyOp = rewriter.create<linalg::CopyOp>(
        loc, TypeRange{tensorType}, ValueRange{tensor}, ValueRange{empty});

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, copyOp.getResult(0),
                                                   ValueRange{zero});

    return success();
  }
};

template <typename ScalarGlobalStoreOp>
struct ScalarGlobalStoreRewrite : public OpRewritePattern<ScalarGlobalStoreOp> {
  using OpRewritePattern<ScalarGlobalStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScalarGlobalStoreOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement memory space check
    if (triton::isOperandMemorySpaceSPM(op.getMemref()))
      return failure();

    auto indices = op.getIndices();
    if (indices.size() > 1) {
      return rewriter.notifyMatchFailure(op, "StoreOp has multiple indices");
    }

    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto index = indices.empty() ? zero : indices[0];

    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);

    auto dst = rewriter.create<memref::SubViewOp>(
        op.getLoc(), op.getMemref(), ValueRange{index}, ValueRange(one),
        ValueRange(one));

    auto val = op.getValue();
    if (!isa<RankedTensorType>(val.getType())) {
      val = rewriter.create<tensor::FromElementsOp>(op.getLoc(), val);
    }
    auto storeOp =
        rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
            op, val, dst);
    storeOp.setWritable(true);

    return success();
  }
};

Operation *findOutmostLoopOp(scf::ForOp forOp) {

  if (!forOp->hasOneUse())
    return forOp;
  auto user = *forOp->getUsers().begin();

  if (!isa<scf::YieldOp>(user))
    return forOp;
  auto parentOp = user->getParentOp();

  if (isa<scf::ForOp>(parentOp))
    return findOutmostLoopOp(cast<scf::ForOp>(parentOp));
  else
    return forOp;
}

Operation *findLoopInsertPoint(Operation *atomicOp) {
  if (!atomicOp->hasOneUse())
    return atomicOp;

  auto user = *atomicOp->getUsers().begin();

  if (!isa<tensor::ExtractOp>(user) && !user->hasOneUse())
    return atomicOp;
  user = *user->getUsers().begin();

  // Whether has mask
  if (isa<scf::YieldOp>(user)) {
    auto parentOp = user->getParentOp();
    if (!isa<scf::IfOp>(parentOp) || !parentOp->hasOneUse())
      return atomicOp;
    user = *parentOp->getUsers().begin();
  }

  if (!isa<tensor::InsertOp>(user) && !user->hasOneUse())
    return atomicOp;
  user = *user->getUsers().begin();

  if (!isa<scf::YieldOp>(user))
    return atomicOp;
  auto parentOp = user->getParentOp();

  if (!isa<scf::ForOp>(parentOp)) {
    return atomicOp;
  }

  return findOutmostLoopOp(cast<scf::ForOp>(parentOp));
}

struct AtomicRMWOpRewrite : public OpRewritePattern<mk::AtomicRMWOp> {
  using OpRewritePattern<mk::AtomicRMWOp>::OpRewritePattern;

  Value createAtomicArithOp(OpBuilder &rewriter, Location loc, Value oldData,
                            Value val, RMWOp rmwOp) const {

    auto inputType = cast<RankedTensorType>(val.getType());
    auto elementType = inputType.getElementType();

    switch (rmwOp) {
    case RMWOp::AND:
      return buildLinalgElementwise<arith::AndIOp>(rewriter, loc,
                                                   ValueRange{val, oldData});
    case RMWOp::OR:
      return buildLinalgElementwise<arith::OrIOp>(rewriter, loc,
                                                  ValueRange{val, oldData});
    case RMWOp::XOR:
      return buildLinalgElementwise<arith::XOrIOp>(rewriter, loc,
                                                   ValueRange{val, oldData});
    case RMWOp::ADD:
      return buildLinalgElementwise<arith::AddIOp>(rewriter, loc,
                                                   ValueRange{val, oldData});
    case RMWOp::FADD:
      return buildLinalgElementwise<arith::AddFOp>(rewriter, loc,
                                                   ValueRange{val, oldData});
    case RMWOp::MAX:
      return elementType.isIntOrIndex()
                 ? buildLinalgElementwise<arith::MaxSIOp>(
                       rewriter, loc, ValueRange{val, oldData})
                 : buildLinalgElementwise<arith::MaximumFOp>(
                       rewriter, loc, ValueRange{val, oldData});
    case RMWOp::UMAX:
      return buildLinalgElementwise<arith::MaxUIOp>(rewriter, loc,
                                                    ValueRange{val, oldData});
    case RMWOp::MIN:
      return elementType.isIntOrIndex()
                 ? buildLinalgElementwise<arith::MinSIOp>(
                       rewriter, loc, ValueRange{val, oldData})
                 : buildLinalgElementwise<arith::MinimumFOp>(
                       rewriter, loc, ValueRange{val, oldData});

    case RMWOp::UMIN:
      return buildLinalgElementwise<arith::MinUIOp>(rewriter, loc,
                                                    ValueRange{val, oldData});
    case RMWOp::XCHG:
      return val;
    default:
      llvm_unreachable("Unexpected atomic op");
    }
  }

  LogicalResult matchAndRewrite(mk::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto val = op.getVal();
    auto inputType = dyn_cast<RankedTensorType>(val.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    auto loc = op.getLoc();
    auto ptr = op.getPtr();

    auto loopInsertPoint = findLoopInsertPoint(op);
    auto insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(loopInsertPoint);
    rewriter.create<mk::AtomicBarrierInOp>(loc);
    rewriter.restoreInsertionPoint(insertionPoint);
    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
        loc, ptr, true /* restrict */, true /* writable */);
    // Read oldData
    auto oldData = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),
                                                    inputType.getElementType());
    auto ddrToSpm =
        rewriter
            .create<linalg::CopyOp>(loc, TypeRange{inputType},
                                    ValueRange{toTensorOp}, ValueRange{oldData})
            ->getResult(0);

    auto newData =
        createAtomicArithOp(rewriter, loc, ddrToSpm, val, op.getAtomicRmwOp());

    auto spmToDDR = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, newData, ptr);
    spmToDDR.setWritable(true);

    rewriter.setInsertionPointAfter(loopInsertPoint);
    rewriter.create<mk::AtomicBarrierOutOp>(loc);
    rewriter.restoreInsertionPoint(insertionPoint);

    rewriter.replaceOp(op, ddrToSpm);

    return success();
  }
};

struct AtomicCASOpRewrite : public OpRewritePattern<mk::AtomicCASOp> {
  using OpRewritePattern<mk::AtomicCASOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mk::AtomicCASOp op,
                                PatternRewriter &rewriter) const override {

    auto val = op.getVal();
    auto inputType = dyn_cast<RankedTensorType>(val.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    auto loc = op.getLoc();
    auto ptr = op.getPtr();

    auto loopInsertPoint = findLoopInsertPoint(op);
    auto insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(loopInsertPoint);
    rewriter.create<mk::AtomicBarrierInOp>(loc);
    rewriter.restoreInsertionPoint(insertionPoint);
    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
        loc, ptr, true /* restrict */, true /* writable */);
    auto empty = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),
                                                  inputType.getElementType());

    // Read oldData
    auto ddrToSpm =
        rewriter
            .create<linalg::CopyOp>(loc, TypeRange{inputType},
                                    ValueRange{toTensorOp}, ValueRange{empty})
            ->getResult(0);

    // compare $cmp with data $old at location $ptr,
    Value cmp = op.getCmp();

    int rank = inputType.getRank();
    auto conditionType =
        RankedTensorType::get(inputType.getShape(), rewriter.getI1Type());
    auto i1Empty = rewriter.create<tensor::EmptyOp>(
        loc, conditionType.getShape(), conditionType.getElementType());
    SmallVector<AffineMap, 3> binaryIndexingMaps(
        3, rewriter.getMultiDimIdentityMap(rank));

    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    Value condition =
        inputType.getElementType().isIntOrIndex()
            ? rewriter
                  .create<linalg::GenericOp>(
                      loc, TypeRange{conditionType}, ValueRange{ddrToSpm, cmp},
                      ValueRange{i1Empty}, binaryIndexingMaps, iteratorTypes,
                      [&](OpBuilder &b, Location loc, ValueRange args) {
                        Value result = b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, args[0], args[1]);
                        b.create<linalg::YieldOp>(loc, result);
                      })
                  ->getResult(0)
            : rewriter
                  .create<linalg::GenericOp>(
                      loc, TypeRange{conditionType}, ValueRange{ddrToSpm, cmp},
                      ValueRange{i1Empty}, binaryIndexingMaps, iteratorTypes,
                      [&](OpBuilder &b, Location loc, ValueRange args) {
                        Value val = b.create<arith::CmpFOp>(
                            loc, arith::CmpFPredicate::OEQ, args[0], args[1]);
                        b.create<linalg::YieldOp>(loc, val);
                      })
                  ->getResult(0);

    SmallVector<AffineMap, 4> selectOpIndexingMaps(
        4, rewriter.getMultiDimIdentityMap(rank));

    auto newData =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{inputType}, ValueRange{condition, val, ddrToSpm},
                ValueRange{empty}, selectOpIndexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value result =
                      b.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
                  b.create<linalg::YieldOp>(loc, result);
                })
            ->getResult(0);

    auto spmToDDR = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, newData, ptr);
    spmToDDR.setWritable(true);

    rewriter.setInsertionPointAfter(loopInsertPoint);
    rewriter.create<mk::AtomicBarrierOutOp>(loc);
    rewriter.restoreInsertionPoint(insertionPoint);

    rewriter.replaceOp(op, ddrToSpm);
    return success();
  }
};

struct BroadcastOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult handleLastDimensionBroadcast(linalg::GenericOp op,
                                             PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto inputs = op.getInputs();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    int rank = inputType.getRank();

    // Build loop bounds for all dimensions except the last one
    SmallVector<int64_t> shapeWithoutLast(inputType.getShape());
    shapeWithoutLast.pop_back();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lbs, ubs, steps;
    for (int64_t size : shapeWithoutLast) {
      lbs.push_back(zero);
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, size));
      steps.push_back(one);
    }

    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, op.getOutputs(),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange indices,
            ValueRange iterArgs) {
          SmallVector<Value> regionInputs(op.getNumDpsInputs());
          SmallVector<Value> inputIndices = indices;
          inputIndices.push_back(zero);
          std::transform(inputs.begin(), inputs.end(), regionInputs.begin(),
                         [&](auto val) {
                           return rewriter.create<tensor::ExtractOp>(
                               loc, val, inputIndices);
                         });

          SmallVector<OpFoldResult> outputOffsets = indices;
          outputOffsets.push_back(rewriter.getIndexAttr(0));
          SmallVector<OpFoldResult> outputSizes(rank - 1,
                                                rewriter.getIndexAttr(1));
          outputSizes.push_back(
              rewriter.getIndexAttr(outputType.getDimSize(rank - 1)));
          SmallVector<OpFoldResult> outputStrides(rank,
                                                  rewriter.getIndexAttr(1));

          Value outSlice = nestedBuilder.create<tensor::ExtractSliceOp>(
              nestedLoc, iterArgs[0], outputOffsets, outputSizes,
              outputStrides);

          Value filled =
              nestedBuilder
                  .create<linalg::FillOp>(nestedLoc, outSlice.getType(),
                                          regionInputs, ValueRange{outSlice})
                  ->getResults()[0];

          auto outValTensor = nestedBuilder.create<tensor::InsertSliceOp>(
              loc, filled, iterArgs[0], outputOffsets, outputSizes,
              outputStrides);
          return SmallVector<Value>{outValTensor};
        });

    rewriter.replaceOp(op, loopNest.results);
    return success();
  }

  std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>>
  initializeSliceParams(RankedTensorType outputType, int rank) const {
    SmallVector<int64_t> sliceOffsets(rank, 0);
    SmallVector<int64_t> sliceSizes(outputType.getShape().begin(),
                                    outputType.getShape().end());
    SmallVector<int64_t> sliceStrides(rank, 1);
    return {sliceOffsets, sliceSizes, sliceStrides};
  }

  memref::AllocOp createOutputMemrefAndInitCopy(linalg::GenericOp op,
                                                PatternRewriter &rewriter,
                                                int64_t broadcastDim) const {
    Location loc = op->getLoc();
    auto inputs = op.getInputs();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    int rank = outputType.getRank();

    auto [sliceOffsets, sliceSizes, sliceStrides] =
        initializeSliceParams(outputType, rank);
    sliceSizes[broadcastDim] =
        1; // Single element slice along broadcast dimension

    // WORKAROUND: For broadcast operations with multiple users, we need to
    // explicitly allocate a new memref to avoid read-after-write conflicts in
    // the bufferized representation. The one-shot-bufferize pass cannot
    // automatically handle these conflicts for memref types in broadcast
    // patterns, so we do it manually here.
    // TODO : In the case where there are multiple users or where there is only
    // one user, we can optimize memory usage by using buffer reuse analysis to
    // reuse memory from output buffers.
    auto outputMemref = rewriter.create<memref::AllocOp>(
        loc,
        MemRefType::get(outputType.getShape(), outputType.getElementType()));
    auto inputMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(inputType.getShape(), inputType.getElementType()),
        inputs[0]);
    auto initSlice =
        rewriter.create<memref::SubViewOp>(loc, outputMemref, sliceOffsets,
                                           /*sizes=*/sliceSizes,
                                           /*strides=*/sliceStrides);
    rewriter.create<memref::CopyOp>(loc, inputMemref, initSlice);
    return outputMemref;
  }

  Value copyWithTilingStrategy(linalg::GenericOp op, PatternRewriter &rewriter,
                               memref::AllocOp outputMemref,
                               int64_t broadcastDim) const {
    Location loc = op->getLoc();
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    int rank = outputType.getRank();
    auto [sliceOffsets, sliceSizes, sliceStrides] =
        initializeSliceParams(outputType, rank);

    int64_t broadcastDimSize = outputType.getShape()[broadcastDim];
    sliceSizes[broadcastDim] = 1;

    // WORKAROUND: Since one-shot-bufferize will insert extra copy for
    // extract_slice and insert_slice even they are different slices. Here we
    // directly use memref copy to avoid extra copy.
    // Copy (1, 2, 4, 8, 16, 32) input slices to output
    constexpr int64_t kTileSizes[] = {1, 2, 4, 8, 16, 32};

    SmallVector<int64_t> currentOffsets = sliceOffsets;
    SmallVector<int64_t> currentSizes = sliceSizes;

    for (int64_t tileSize : kTileSizes) {
      if (tileSize >= broadcastDimSize)
        break;

      currentSizes[broadcastDim] = tileSize;
      auto sourceSlice = rewriter.create<memref::SubViewOp>(
          loc, outputMemref, sliceOffsets, currentSizes, sliceStrides);

      currentOffsets[broadcastDim] = tileSize;
      auto destSlice = rewriter.create<memref::SubViewOp>(
          loc, outputMemref, currentOffsets, currentSizes, sliceStrides);

      rewriter.create<memref::CopyOp>(loc, sourceSlice, destSlice);
    }
    return rewriter.create<bufferization::ToTensorOp>(
        loc, outputType, outputMemref, /*allow_memref_to_tensor=*/true,
        /*allow_tensor_to_memref=*/true);
  }

  LogicalResult handleLargeBroadcastCase(linalg::GenericOp op,
                                         PatternRewriter &rewriter,
                                         Value sourceTensor,
                                         int64_t broadcastDim) const {
    Location loc = op->getLoc();
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    int rank = outputType.getRank();
    int64_t broadcastDimSize = outputType.getShape()[broadcastDim];
    auto [sliceOffsets, sliceSizes, sliceStrides] =
        initializeSliceParams(outputType, rank);

    SmallVector<int64_t> largeSliceShape = sliceSizes;
    largeSliceShape[broadcastDim] = kMaxSliceSize;
    auto largeSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc,
        RankedTensorType::get(largeSliceShape, outputType.getElementType()),
        sourceTensor, ValueRange(), ValueRange(), ValueRange(), sliceOffsets,
        largeSliceShape, sliceStrides);

    Value lowerBound =
        rewriter.create<arith::ConstantIndexOp>(loc, kMaxSliceSize);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, broadcastDimSize);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, kMaxSliceSize);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{sourceTensor},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange iterArgs) {
          SmallVector<OpFoldResult> outputOffsets(rank,
                                                  rewriter.getIndexAttr(0));
          outputOffsets[broadcastDim] = iv;
          SmallVector<OpFoldResult> outputSizes;
          for (auto s : outputType.getShape())
            outputSizes.push_back(rewriter.getIndexAttr(s));
          outputSizes[broadcastDim] = rewriter.getIndexAttr(kMaxSliceSize);
          SmallVector<OpFoldResult> outputStrides(rank,
                                                  rewriter.getIndexAttr(1));
          auto outputSlice = nestedBuilder.create<tensor::InsertSliceOp>(
              nestedLoc, largeSlice, iterArgs[0], outputOffsets, outputSizes,
              outputStrides);
          nestedBuilder.create<scf::YieldOp>(nestedLoc,
                                             outputSlice.getResult());
        });

    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }

  LogicalResult handleOtherDimensionBroadcast(linalg::GenericOp op,
                                              PatternRewriter &rewriter,
                                              int64_t broadcastDim) const {
    Location loc = op->getLoc();
    auto inputs = op.getInputs();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    int rank = inputType.getRank();
    int64_t broadcastDimSize = outputType.getShape()[broadcastDim];

    auto outputMemref =
        createOutputMemrefAndInitCopy(op, rewriter, broadcastDim);
    auto resultTensor =
        copyWithTilingStrategy(op, rewriter, outputMemref, broadcastDim);

    if (broadcastDimSize <= kMaxSliceSize) {
      rewriter.replaceAllOpUsesWith(op, resultTensor);
      return success();
    }

    return handleLargeBroadcastCase(op, rewriter, resultTensor,
                                                  broadcastDim);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (!regionOps.empty() || !op->hasAttr("broadcastDims"))
      return failure();

    auto dims = cast<DenseI64ArrayAttr>(op->getAttr("broadcastDims"));
    auto inputs = op.getInputs();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto rank = inputType.getRank();
    if (dims.size() != 1)
      return failure();

    if (dims[0] == rank - 1) {
      return handleLastDimensionBroadcast(op, rewriter);
    } else {
      return handleOtherDimensionBroadcast(op, rewriter, dims[0]);
    }
  }

private:
  constexpr static int64_t kMaxSliceSize = 64;
};

struct DivFloatOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult convertDivFloatOp(linalg::GenericOp op,
                                  PatternRewriter &rewriter) const {

    Location loc = op->getLoc();

    auto inputTensorType =
        dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rank = inputTensorType.getRank();
    auto outputTensorType =
        dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, inputTensorType.getShape(), inputTensorType.getElementType());

    Value recip = rewriter
                      .create<linalg::ReciprocalOp>(
                          loc, inputTensorType, ValueRange{op.getInputs()[1]},
                          ValueRange{empty})
                      ->getResult(0);

    SmallVector<AffineMap, 3> binaryIndexingMaps(
        3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, outputTensorType, ValueRange{op.getInputs()[0], recip},
        ValueRange{empty}, binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 || !isa<arith::DivFOp>(regionOps[0]))
      return failure();

    return convertDivFloatOp(op, rewriter);
  }
};

struct DivIntOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 ||
        !isa<arith::DivSIOp, arith::DivUIOp>(regionOps[0]))
      return failure();

    // FIXME: Canonicalize non-precision mode divint in linalg-to-mk, others
    // default to scf.for
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult(0).getType());
    SmallVector<Value> inputs(op.getInputs().begin(), op.getInputs().end());

    SmallVector<Value> outputs = {rewriter.create<tensor::EmptyOp>(
        op->getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType())};
    assert(op->getResultTypes().size() == 1);

    auto scalarResultType = resultTensorType.getElementType();

    // NOTE: linalgOpToloop function only support memref type
    auto shape = resultTensorType.getShape();
    auto loc = op->getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lbs, ubs, steps;
    for (auto [i, size] : enumerate(shape)) {
      auto sizeValue = rewriter.create<arith::ConstantIndexOp>(loc, size);
      lbs.push_back(zero);
      ubs.push_back(sizeValue);
      steps.push_back(one);
    }
    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, outputs,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange indices,
            ValueRange iterArgs) {
          SmallVector<Value> regionInputs(op.getNumDpsInputs());
          std::transform(inputs.begin(), inputs.end(), regionInputs.begin(),
                         [&](auto val) {
                           return rewriter.create<tensor::ExtractOp>(loc, val,
                                                                     indices);
                         });
          auto *scalarOp = nestedBuilder.create(
              loc, regionOps[0]->getName().getIdentifier(), regionInputs,
              scalarResultType, regionOps[0]->getAttrs());

          auto outValTensor = nestedBuilder.create<tensor::InsertOp>(
              loc, scalarOp->getResult(0), iterArgs[0], indices);
          return SmallVector<Value>{outValTensor};
        });

    rewriter.replaceOp(op, loopNest.results);
    return success();
  }
};

struct SelectOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult convertI1Select(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto mask = op.getInputs()[0];
    auto input1 = op.getInputs()[1];
    auto input2 = op.getInputs()[2];

    auto inputType = dyn_cast<RankedTensorType>(input1.getType());
    auto rank = inputType.getRank();

    auto empty = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),
                                                  inputType.getElementType());

    SmallVector<AffineMap, 3> binaryIndexingMaps(
        3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    // result[i] = (mask[i] AND A[i]) OR (NOT mask[i] AND B[i])
    // mask[i] AND A[i]
    auto lhs = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{mask, input1}, ValueRange{empty},
        binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::AndIOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    // NOT mask
    // NOTE: Also can define a mk.not
    Value allTrue = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
    allTrue =
        rewriter
            .create<linalg::FillOp>(loc,
                                    ValueRange{
                                        rewriter.create<arith::ConstantOp>(
                                            loc, inputType.getElementType(),
                                            rewriter.getBoolAttr(true)),
                                    },
                                    ValueRange{empty})
            ->getResult(0);
    auto notMask = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{mask, allTrue}, ValueRange{empty},
        binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::XOrIOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    // NOT mask[i] AND B[i]
    auto rhs = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{notMask.getResult(0), input2},
        ValueRange{empty}, binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::AndIOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    // result
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, inputType, ValueRange{lhs.getResult(0), rhs.getResult(0)},
        ValueRange{empty}, binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::OrIOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    return success();
  }

  LogicalResult convertI8Select(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto mask = op.getInputs()[0];
    auto input1 = op.getInputs()[1];
    auto input2 = op.getInputs()[2];
    auto inputType = dyn_cast<RankedTensorType>(input1.getType());
    auto midType =
        RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
    auto fpinput1 =
        buildLinalgElementwise<arith::SIToFPOp>(rewriter, loc, midType, input1);
    auto fpinput2 =
        buildLinalgElementwise<arith::SIToFPOp>(rewriter, loc, midType, input2);
    auto fpOutput = buildLinalgElementwise<arith::SelectOp>(
        rewriter, loc, midType, {mask, fpinput1, fpinput2});
    auto i8Output = buildLinalgElementwise<arith::FPToSIOp>(
        rewriter, loc, inputType, fpOutput);
    rewriter.replaceAllUsesWith(op->getResults(), {i8Output});
    rewriter.eraseOp(op);
    return success();
  }

  Value createBitcastOp(OpBuilder &rewriter, Location loc, Value input,
                        RankedTensorType targetType) const {
    auto empty = rewriter.create<tensor::EmptyOp>(loc, targetType.getShape(),
                                                  targetType.getElementType());
    int rank = targetType.getRank();

    SmallVector<AffineMap, 2> binaryIndexingMaps(
        2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);

    return rewriter
        .create<linalg::GenericOp>(
            loc, targetType, ValueRange{input}, ValueRange{empty},
            binaryIndexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value result = b.create<arith::BitcastOp>(
                  loc, targetType.getElementType(), args[0]);
              b.create<linalg::YieldOp>(loc, result);
            })
        ->getResult(0);
  }

  LogicalResult SelectConvertOp(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto mask = op.getInputs()[0];
    auto input1 = op.getInputs()[1];
    auto input2 = op.getInputs()[2];

    auto inputType = dyn_cast<RankedTensorType>(input1.getType());
    auto rank = inputType.getRank();

    auto castType = inputType;
    if (inputType.getElementType().isIntOrIndex()) {
      auto bitWidth = inputType.getElementTypeBitWidth();
      assert(bitWidth == 16 || bitWidth == 32);
      FloatType floatType =
          bitWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
      castType = RankedTensorType::get(inputType.getShape(), floatType);
      // TODO: Bitcast integer to float
      input1 = createBitcastOp(rewriter, op.getLoc(), input1, castType);
      input2 = createBitcastOp(rewriter, op.getLoc(), input2, castType);
    }

    auto empty = rewriter.create<tensor::EmptyOp>(loc, castType.getShape(),
                                                  castType.getElementType());

    // Maskmove mask only support int8/fp, here mask is memref<i1>
    auto maskCast =
        rewriter.create<mk::Bit2FpOp>(op.getLoc(), castType, mask, empty);

    // Res = input2;
    Value res = rewriter
                    .create<linalg::CopyOp>(loc, castType, ValueRange{input2},
                                            ValueRange{empty})
                    ->getResult(0);

    // if input0 = 1, Res = input1;
    // if input0 = 0, Res = input2;
    res = rewriter
              .create<mk::MaskMoveOp>(loc, castType, input1,
                                      maskCast->getResult(0), res)
              ->getResult(0);

    if (inputType != castType) {
      res = createBitcastOp(rewriter, op.getLoc(), res, inputType);
    }
    rewriter.replaceOp(op, res);
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 || !isa<arith::SelectOp>(regionOps[0]))
      return failure();

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    assert(inputType && "Only support ranked tensor type");
    auto elemType = inputType.getElementType();
    if (elemType.getIntOrFloatBitWidth() == 64)
      return failure();
    if (elemType.isInteger(1))
      return convertI1Select(op, rewriter);
    // maskmove does not support int8, so convert to fp
    if (elemType.isInteger(8))
      return convertI8Select(op, rewriter);

    return SelectConvertOp(op, rewriter);
  }
};

struct IsInfOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult convertIsInfOp(linalg::GenericOp op,
                               PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    auto inputTensorType =
        dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outputTensorType =
        dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, inputTensorType.getShape(), inputTensorType.getElementType());
    // 1 / inf == 0, Use recip and boolequalvs to calculate isinf.
    Value recip =
        rewriter
            .create<linalg::ReciprocalOp>(loc, inputTensorType, op.getInputs(),
                                          ValueRange{empty})
            ->getResult(0);

    auto zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(inputTensorType.getElementType()));
    rewriter.replaceOpWithNewOp<mk::BoolEqualVS>(op, outputTensorType, recip,
                                                 zero, op.getOutputs()[0]);
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 || !isa<math::IsInfOp>(regionOps[0]))
      return failure();

    return convertIsInfOp(op, rewriter);
  }
};

struct PowFOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  Value computeOddFloatIndicator(PatternRewriter &rewriter, Value exponent,
                                 Location loc) const {
    auto resultType = cast<RankedTensorType>(exponent.getType());
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    auto zeroTensor = rewriter
                          .create<linalg::FillOp>(loc, ValueRange{zeroVal},
                                                  ValueRange{emptyTensor})
                          ->getResult(0);

    auto oneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(resultType.getElementType()));
    auto oneTensor = rewriter
                         .create<linalg::FillOp>(loc, ValueRange{oneVal},
                                                 ValueRange{emptyTensor})
                         ->getResult(0);

    auto twoVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultType.getElementType(), 2.0));
    auto twoTensor = rewriter
                         .create<linalg::FillOp>(loc, ValueRange{twoVal},
                                                 ValueRange{emptyTensor})
                         ->getResult(0);

    // Calculate exponent % 2: returns 1 if odd, 0 if even
    auto remainder = buildLinalgElementwise<arith::RemFOp>(
        rewriter, loc, resultType, ValueRange{exponent, twoTensor});

    return rewriter
        .create<mk::MaskMoveOp>(loc, resultType, oneTensor, remainder,
                                zeroTensor)
        ->getResult(0);
  }

  Value computeNegativeResultIndicator(PatternRewriter &rewriter, Location loc,
                                       Value base, Value exponent,
                                       Value &absoluteBase) const {
    auto resultType = cast<RankedTensorType>(base.getType());
    auto intResultType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI32Type());

    auto zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Check if base is negative (a < 0)
    auto isBaseNegative =
        rewriter
            .create<mk::LessThenVS>(loc, resultType, base, zeroVal, emptyTensor)
            ->getResult(0);

    // Check if exponent is integer-like (truncated value equals original)
    auto truncatedExponent = buildLinalgElementwise<math::TruncOp>(
        rewriter, loc, resultType, ValueRange{exponent});
    auto isIntegerLikeExponent =
        rewriter
            .create<mk::EqualVV>(loc, resultType, exponent, truncatedExponent,
                                 emptyTensor)
            ->getResult(0);

    // Compute absolute base for integer-like exponents : a = |a|
    auto absoluteBaseValue = buildLinalgElementwise<math::AbsFOp>(
        rewriter, loc, resultType, ValueRange{base});
    absoluteBase =
        rewriter
            .create<mk::MaskMoveOp>(loc, resultType, absoluteBaseValue,
                                    isIntegerLikeExponent, base)
            ->getResult(0);

    // Convert boolean masks to integer for bitwise operations
    auto isBaseNegativeInt = buildLinalgElementwise<arith::BitcastOp>(
        rewriter, loc, intResultType, ValueRange{isBaseNegative});
    auto isIntegerLikeExponentInt = buildLinalgElementwise<arith::BitcastOp>(
        rewriter, loc, intResultType, ValueRange{isIntegerLikeExponent});
    auto canTakeAbsolute = buildLinalgElementwise<arith::AndIOp>(
        rewriter, loc, intResultType,
        ValueRange{isBaseNegativeInt, isIntegerLikeExponentInt});

    // Check if integer-like exponent is odd : b % 2 != 0
    auto isOddIndicator =
        computeOddFloatIndicator(rewriter, truncatedExponent, loc);
    auto isOddIndicatorInt = buildLinalgElementwise<arith::BitcastOp>(
        rewriter, loc, intResultType, ValueRange{isOddIndicator});

    // Final condition: a < 0 & b is IntergerLike & b % 2 != 0
    auto isResultNegative = buildLinalgElementwise<arith::AndIOp>(
        rewriter, loc, intResultType,
        ValueRange{canTakeAbsolute, isOddIndicatorInt});

    return buildLinalgElementwise<arith::BitcastOp>(
        rewriter, loc, resultType, ValueRange{isResultNegative});
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 || !isa<math::PowFOp>(regionOps.front()))
      return failure();

    auto base = op.getInputs()[0];
    auto exponent = op.getInputs()[1];
    auto loc = op->getLoc();
    auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);

    auto zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    Value absoluteBase;
    // Determine if result should be negative:
    // (a <0 && b isIntegerLike && b % 2!= 0)
    auto isResultNegative = computeNegativeResultIndicator(
        rewriter, loc, base, exponent, absoluteBase);

    // Compute power using identity: a^b = 2^(b * log2(|a|))
    auto log2Base = buildLinalgElementwise<math::Log2Op>(
        rewriter, loc, resultType, ValueRange{absoluteBase});
    auto exponentTimesLog = buildLinalgElementwise<arith::MulFOp>(
        rewriter, loc, resultType, ValueRange{log2Base, exponent});
    auto powerResult = buildLinalgElementwise<math::Exp2Op>(
        rewriter, loc, resultType, ValueRange{exponentTimesLog});

    // Apply negative sign if isResultNegative: a ^ b => -2 ^ (b * log2(|a|))
    auto negativePowerResult = buildLinalgElementwise<arith::NegFOp>(
        rewriter, loc, resultType, ValueRange{powerResult});
    auto signedResult =
        rewriter
            .create<mk::MaskMoveOp>(loc, resultType, negativePowerResult,
                                    isResultNegative, powerResult)
            ->getResult(0);

    // Handle special case: if exponent = 0 , a ^ b = 1
    auto isZeroExponent = rewriter
                              .create<mk::EqualVS>(loc, resultType, exponent,
                                                   zeroVal, emptyTensor)
                              ->getResult(0);
    auto oneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(resultType.getElementType()));
    auto oneTensor = rewriter
                         .create<linalg::FillOp>(loc, ValueRange{oneVal},
                                                 ValueRange{emptyTensor})
                         ->getResult(0);

    auto finalResult = rewriter.create<mk::MaskMoveOp>(
        loc, resultType, oneTensor, isZeroExponent, signedResult);

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

struct MinMaxOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  template <typename MKOpT>
  LogicalResult convertMinMaxOp(linalg::GenericOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];

    auto inputType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rank = inputType.getRank();

    auto inputTypeEmpty = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());

    SmallVector<AffineMap, 3> binaryIndexingMaps(
        3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // auto isANan = UnEqualVV(lhs, lhs)
    auto isANan = rewriter.create<mk::UnEqualVV>(loc, inputType, lhs, lhs,
                                                 inputTypeEmpty);

    // auto result = lhs
    Value res = rewriter
                    .create<linalg::CopyOp>(loc, inputType, ValueRange{lhs},
                                            ValueRange{inputTypeEmpty})
                    ->getResult(0);

    // result = maskmove(isANan, rhs)
    res = rewriter
              .create<mk::MaskMoveOp>(loc, inputType, rhs, isANan->getResult(0),
                                      res)
              ->getResult(0);

    // auto isBNan = UnEqualVV(rhs, rhs)
    auto isBNan =
        rewriter
            .create<mk::UnEqualVV>(loc, inputType, rhs, rhs, inputTypeEmpty)
            ->getResult(0);

    // auto shouldApplyResult = EqualVS(isBNan, 0)
    auto constValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(inputType.getElementType()));
    auto shouldApplyResult = rewriter.create<mk::EqualVS>(
        loc, inputType, isBNan, constValue, inputTypeEmpty);

    // auto minMaxValue = maxvv/minvv(result, rhs)
    auto minMaxValue = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{res, rhs}, ValueRange{inputTypeEmpty},
        binaryIndexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<MKOpT>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    // result = maskmove(shouldApplyResult, minMaxValue)
    res = rewriter
              .create<mk::MaskMoveOp>(loc, inputType, minMaxValue->getResult(0),
                                      shouldApplyResult->getResult(0), res)
              .getResult(0);

    rewriter.replaceOp(op, res);

    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1 ||
        !isa<arith::MaxNumFOp, arith::MinNumFOp>(regionOps[0]))
      return failure();

    if (isa<arith::MaxNumFOp>(regionOps[0]))
      return convertMinMaxOp<arith::MaximumFOp>(op, rewriter);
    else
      return convertMinMaxOp<arith::MinimumFOp>(op, rewriter);
  }
};

template <typename NaryOp>
Value createElemwiseNaryOp(OpBuilder &builder, Location loc, ValueRange inputs,
                           Value output) {
  auto outputTy = cast<RankedTensorType>(output.getType());
  auto rank = outputTy.getRank();
  if (rank == 0) {
    SmallVector<Value> loadVals;
    llvm::transform(inputs, std::back_inserter(loadVals), [&](Value input) {
      return builder.create<tensor::ExtractOp>(loc, input, ValueRange{});
    });
    auto val = builder.create<NaryOp>(loc, outputTy.getElementType(), loadVals);
    return builder.create<tensor::FromElementsOp>(loc, outputTy,
                                                  val.getResult());
  } else {
    SmallVector<AffineMap> idMaps(2, builder.getMultiDimIdentityMap(rank));
    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    return builder
        .create<linalg::GenericOp>(
            loc, outputTy, inputs, ValueRange{output}, idMaps, iterators,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value val = b.create<NaryOp>(loc, args.back().getType(),
                                           args.drop_back());
              b.create<linalg::YieldOp>(loc, val);
            })
        .getResult(0);
  }
}

static LogicalResult convertSIOpToF32Op(
    Operation *srcOp, PatternRewriter &rewriter, ValueRange inputs,
    ValueRange outputs,
    std::function<ValueRange(Operation *srcOp, PatternRewriter &rewrite,
                             ValueRange inputs, ValueRange outputs)>
        fpOpBuildFn) {
  Location loc = srcOp->getLoc();
  SmallVector<Value> fpInputs, fpOutputs, intResults;
  // Convert integer input
  for (auto input : inputs) {
    auto inputTy = cast<RankedTensorType>(input.getType());
    Value fpInput = rewriter.create<tensor::EmptyOp>(loc, inputTy.getShape(),
                                                     rewriter.getF32Type());
    fpInputs.push_back(
        createElemwiseNaryOp<arith::SIToFPOp>(rewriter, loc, input, fpInput));
  }

  for (auto output : outputs) {

    auto outputTy = cast<RankedTensorType>(output.getType());
    Value fpOutput = rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(),
                                                      rewriter.getF32Type());

    fpOutputs.push_back(fpOutput);
  }

  auto fpResults = fpOpBuildFn(srcOp, rewriter, fpInputs, fpOutputs);
  auto resultTy = cast<RankedTensorType>(srcOp->getResultTypes()[0]);
  for (auto fpResult : fpResults) {

    Value intResult = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType());
    intResults.push_back(createElemwiseNaryOp<arith::FPToSIOp>(
        rewriter, loc, fpResult, intResult));
  }
  rewriter.replaceOp(srcOp, intResults);
  return success();
}

struct CannonicalizeRedudantTypeConversion
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isValidSIToFPOp(linalg::GenericOp op, PatternRewriter &rewriter) const {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1)
      return false;
    if (!dyn_cast<arith::SIToFPOp>(regionOps.front()))
      return false;

    auto inputType = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto outputType = cast<RankedTensorType>(op->getResultTypes()[0]);

    return inputType.getElementType() == rewriter.getI64Type() &&
           outputType.getElementType() == rewriter.getF32Type();
  }

  bool isValidFPToSIOp(linalg::GenericOp op, PatternRewriter &rewriter) const {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.empty())
      return false;
    if (!dyn_cast<arith::FPToSIOp>(regionOps[0]))
      return false;

    auto inputType = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto outputType = cast<RankedTensorType>(op->getResultTypes()[0]);

    return inputType.getElementType() == rewriter.getF32Type() &&
           outputType.getElementType() == rewriter.getI64Type();
  }

  bool isBroadcastOp(linalg::GenericOp op) const {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    return regionOps.empty() && op->hasAttr("broadcastDims");
  }

  LogicalResult handleBroadcastCase(linalg::GenericOp op,
                                    linalg::GenericOp broadcastOp,
                                    PatternRewriter &rewriter) const {
    auto broadcastInput = broadcastOp.getInputs()[0];
    auto broadcastInputType =
        cast<RankedTensorType>(broadcastOp->getOperandTypes()[0]);
    auto outputType = cast<RankedTensorType>(op->getResultTypes()[0]);
    auto inputType = cast<RankedTensorType>(op->getOperandTypes()[0]);

    auto newSITOFPType = RankedTensorType::get(broadcastInputType.getShape(),
                                               outputType.getElementType());
    auto newSITOFP =
        createNewSIToFpOp(op, broadcastInput, newSITOFPType, rewriter);
    auto newBroadcast =
        createNewBroadcast(broadcastOp, newSITOFP, outputType, rewriter);
    rewriter.replaceAllOpUsesWith(op, newBroadcast);
    return success();
  }

  Value createNewBroadcast(linalg::GenericOp broadcastOp, Value input,
                           RankedTensorType outputType,
                           PatternRewriter &rewriter) const {

    auto empty = rewriter.create<tensor::EmptyOp>(broadcastOp->getLoc(),
                                                  outputType.getShape(),
                                                  outputType.getElementType());
    auto broadcast = rewriter.create<linalg::GenericOp>(
        broadcastOp->getLoc(), outputType, input, ValueRange{empty},
        broadcastOp.getIndexingMapsArray(), broadcastOp.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args.drop_back());
        });

    broadcast->setAttr("broadcastDims", broadcastOp->getAttr("broadcastDims"));
    return broadcast->getResult(0);
  }

  Value createNewSIToFpOp(linalg::GenericOp originalOp, Value input,
                          RankedTensorType outputType,
                          PatternRewriter &rewriter) const {
    auto empty = rewriter.create<tensor::EmptyOp>(originalOp->getLoc(),
                                                  outputType.getShape(),
                                                  outputType.getElementType());
    auto newOp = rewriter.create<linalg::GenericOp>(
        originalOp->getLoc(), outputType, ValueRange{input}, ValueRange{empty},
        originalOp.getIndexingMapsArray(), originalOp.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          auto sitofp = b.create<arith::SIToFPOp>(loc, args.back().getType(),
                                                  args.drop_back());
          b.create<linalg::YieldOp>(loc, sitofp->getResult(0));
        });

    return newOp->getResult(0);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Todo : support eliminate other type fptosi and sitofp.
    // As far, only support eliminate i64 type fptosi and sitofp(i64/f32).
    if (!isValidSIToFPOp(op, rewriter)) {
      return failure();
    }

    auto input = op.getInputs()[0];
    auto prevOp = input.getDefiningOp<linalg::GenericOp>();
    if (!prevOp) {
      return failure();
    }
    if (isBroadcastOp(prevOp)) {
      return handleBroadcastCase(op, prevOp, rewriter);
    } else if (isValidFPToSIOp(prevOp, rewriter)) {
      rewriter.replaceAllOpUsesWith(op, prevOp.getInputs()[0]);
      return success();
    }

    return failure();
  }
};

struct CastElementwiseOpIOToFloatPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  void initialize() {
    // Register conversions from SIOp to FPOp
    registerSIOpMapFPOp<arith::AddIOp, arith::AddFOp>();
    registerSIOpMapFPOp<arith::SubIOp, arith::SubFOp>();
    registerSIOpMapFPOp<arith::MulIOp, arith::MulFOp>();
    registerSIOpMapFPOp<arith::MaxSIOp, arith::MaximumFOp>();
    registerSIOpMapFPOp<arith::MinSIOp, arith::MinimumFOp>();
    registerSIOpMapFPOp<math::AbsIOp, math::AbsFOp>();
    registerSIOpMapFPOp<arith::RemSIOp, arith::RemFOp>();
  }

  template <typename SIOp, typename FPOp> void registerSIOpMapFPOp() {
    OperationName SIOpName(SIOp::getOperationName(), getContext());
    assert(!SIToFPOpBuildFnMap.contains(SIOpName) &&
           "SIOp already registered for conversion to FPOp");
    SIToFPOpBuildFnMap[SIOpName] =
        [](Operation *srcOp, PatternRewriter &rewriter, ValueRange inputs,
           ValueRange outputs) -> ValueRange {
      auto genericOp = cast<linalg::GenericOp>(srcOp);
      return rewriter
          .create<linalg::GenericOp>(
              srcOp->getLoc(), outputs.back().getType(), inputs, outputs,
              genericOp.getIndexingMapsArray(),
              genericOp.getIteratorTypesArray(),
              [](OpBuilder &b, Location loc, ValueRange args) {
                Value val = b.create<FPOp>(loc, args.back().getType(),
                                           args.drop_back());
                b.create<linalg::YieldOp>(loc, val);
              })
          .getResults();
    };
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1)
      return failure();

    Location loc = op->getLoc();
    auto elemWiseOp = regionOps[0];
    OperationName OpName = elemWiseOp->getName();
    auto inputs = op.getInputs();
    // NOTE: Output not always exist
    auto outputs = op.getOutputs();

    if (SIToFPOpBuildFnMap.contains(OpName) &&
        (!precisionPriority || cast<RankedTensorType>(outputs[0].getType())
                                       .getElementType()
                                       .getIntOrFloatBitWidth() < 32)) {
      assert(outputs.size() == 1 &&
             "Elementwise conversion only support single output");
      assert(cast<RankedTensorType>(outputs[0].getType())
                 .getElementType()
                 .isInteger() &&
             "Output type must be integer type");

      return convertSIOpToF32Op(op, rewriter, op.getInputs(), op.getOutputs(),
                                SIToFPOpBuildFnMap.at(OpName));
    }

    if (auto cmpiOp = dyn_cast<arith::CmpIOp>(elemWiseOp)) {
      auto inputType = cast<RankedTensorType>(inputs.front().getType());

      if (inputType.getNumElements() < 8)
        return failure();

      if (precisionPriority &&
          inputType.getElementType().getIntOrFloatBitWidth() >= 32)
        return failure();

      auto outputType =
          cast<RankedTensorType>(op.getOutputs().front().getType());

      arith::CmpFPredicate fpPred;
      switch (cmpiOp.getPredicate()) {
      default:
        return failure();
      case arith::CmpIPredicate::eq:
        fpPred = arith::CmpFPredicate::OEQ;
        break;
      case arith::CmpIPredicate::ne:
        fpPred = arith::CmpFPredicate::ONE;
        break;
      case arith::CmpIPredicate::sge:
        fpPred = arith::CmpFPredicate::OGE;
        break;
      case arith::CmpIPredicate::sgt:
        fpPred = arith::CmpFPredicate::OGT;
        break;
      case arith::CmpIPredicate::sle:
        fpPred = arith::CmpFPredicate::OLE;
        break;
      case arith::CmpIPredicate::slt:
        fpPred = arith::CmpFPredicate::OLT;
        break;
      }

      return convertSIOpToF32Op(
          op, rewriter, op.getInputs(), ValueRange{},
          [&](Operation *srcOp, PatternRewriter &rewriter, ValueRange inputs,
              ValueRange outputs) {
            auto genericOp = cast<linalg::GenericOp>(srcOp);

            return rewriter
                .create<linalg::GenericOp>(
                    srcOp->getLoc(), outputType, inputs, genericOp.getOutputs(),
                    genericOp.getIndexingMapsArray(),
                    genericOp.getIteratorTypesArray(),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                      Value val = b.create<arith::CmpFOp>(loc, fpPred, args[0],
                                                          args[1]);
                      b.create<linalg::YieldOp>(loc, val);
                    })
                ->getResults();
          });
    }

    if (auto divsiOp = dyn_cast<arith::DivSIOp>(elemWiseOp)) {
      if (precisionPriority)
        return failure();
      return convertSIOpToF32Op(
          op, rewriter, op.getInputs(), op.getOutputs(),
          [&](Operation *srcOp, PatternRewriter &rewriter, ValueRange inputs,
              ValueRange outputs) {
            auto genericOp = cast<linalg::GenericOp>(srcOp);
            auto divf =
                rewriter
                    .create<linalg::GenericOp>(
                        srcOp->getLoc(), outputs.front().getType(), inputs,
                        outputs, genericOp.getIndexingMapsArray(),
                        genericOp.getIteratorTypesArray(),
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value val = b.create<arith::DivFOp>(
                              loc, args.back().getType(), args[0], args[1]);
                          b.create<linalg::YieldOp>(loc, val);
                        })
                    ->getResult(0);
            SmallVector<AffineMap> indexingMaps(
                2, genericOp.getIndexingMapsArray().front());
            return rewriter
                .create<linalg::GenericOp>(
                    srcOp->getLoc(), outputs.front().getType(),
                    ValueRange{divf}, ValueRange{divf}, indexingMaps,
                    genericOp.getIteratorTypesArray(),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                      Value val = b.create<math::TruncOp>(
                          loc, args.back().getType(), args.drop_back());
                      b.create<linalg::YieldOp>(loc, val);
                    })
                ->getResults();
          });
    }

    return failure();
  }

  CastElementwiseOpIOToFloatPattern(MLIRContext *context,
                                    bool precisionPriority)
      : OpRewritePattern<linalg::GenericOp>(context),
        precisionPriority(precisionPriority) {}

private:
  // Map from SIOp to FPOp conversion functions
  llvm::DenseMap<OperationName,
                 std::function<ValueRange(Operation *, PatternRewriter &,
                                          ValueRange, ValueRange)>>
      SIToFPOpBuildFnMap;

  bool precisionPriority = false;
};

struct CastReduceOpIOToFloatPattern
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  void initialize() {
    // Register conversions from SIOp to FPOp
    registerSIOpMapFPOp<arith::AddIOp, arith::AddFOp>();
    registerSIOpMapFPOp<arith::MulIOp, arith::MulFOp>();
    registerSIOpMapFPOp<arith::MaxSIOp, arith::MaximumFOp>();
    registerSIOpMapFPOp<arith::MinSIOp, arith::MinimumFOp>();
  }

  template <typename SIOp, typename FPOp> void registerSIOpMapFPOp() {
    OperationName SIOpName(SIOp::getOperationName(), getContext());
    assert(!SIToFPOpBuildFnMap.contains(SIOpName) &&
           "SIOp already registered for conversion to FPOp");
    SIToFPOpBuildFnMap[SIOpName] = [](Operation *op, PatternRewriter &rewriter,
                                      ValueRange inputs,
                                      ValueRange outputs) -> ValueRange {
      auto reduceOp = cast<linalg::ReduceOp>(op);
      return rewriter
          .create<linalg::ReduceOp>(
              reduceOp->getLoc(), inputs, outputs, reduceOp.getDimensions(),
              [](OpBuilder &b, Location loc, ValueRange args) {
                Value val = b.create<FPOp>(loc, args.back().getType(), args);
                b.create<linalg::YieldOp>(loc, val);
              })
          .getResults();
    };
  }

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::ReduceOp>(op);
    if (regionOps.size() != 1)
      return failure();

    auto reduceOp = regionOps[0];
    OperationName OpName = reduceOp->getName();

    if (SIToFPOpBuildFnMap.contains(OpName) &&
        (!precisionPriority ||
         cast<RankedTensorType>(op.getInits().front().getType())
                 .getElementType()
                 .getIntOrFloatBitWidth() < 32)) {

      assert(op.getInits().size() == 1 &&
             "Reduce conversion only support single output");

      auto constantType =
          cast<RankedTensorType>(op.getInits().front().getType())
              .getElementType();

      auto attr = getRedBaseAttr(rewriter, reduceOp, constantType);
      if (checkReductionBaseAttr(op, rewriter, attr))
        return rewriter.notifyMatchFailure(
            op, "Reduction op has invalid init value");

      return convertSIOpToF32Op(op, rewriter, op.getInputs(), op.getInits(),
                                SIToFPOpBuildFnMap.at(OpName));
    }

    return failure();
  }

  CastReduceOpIOToFloatPattern(MLIRContext *context, bool precisionPriority)
      : OpRewritePattern<linalg::ReduceOp>(context),
        precisionPriority(precisionPriority) {}

private:
  // Map from SIOp to FPOp conversion functions
  llvm::DenseMap<OperationName,
                 std::function<ValueRange(Operation *, PatternRewriter &,
                                          ValueRange, ValueRange)>>
      SIToFPOpBuildFnMap;

  bool precisionPriority = false;
};

template <typename MKOpT>
struct CastArgMinMaxOpIOToFloatPattern : public OpRewritePattern<MKOpT> {
  using OpRewritePattern<MKOpT>::OpRewritePattern;
  using OpAdaptor = typename MKOpT::Adaptor;
  LogicalResult matchAndRewrite(MKOpT op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getSrc();
    auto inputTy = cast<RankedTensorType>(input.getType());
    if (!inputTy.getElementType().isInteger()) {
      return failure();
    }

    auto bitWidth = inputTy.getElementTypeBitWidth();
    if (bitWidth != 16 && bitWidth != 32 && bitWidth != 64) {
      return failure();
    }
    auto loc = op->getLoc();

    Value inputEmpty = rewriter.create<tensor::EmptyOp>(loc, inputTy.getShape(),
                                                        rewriter.getF32Type());
    Value fpInput = createElemwiseNaryOp<arith::SIToFPOp>(rewriter, loc,
                                                          {input}, inputEmpty);
    // FIXME: Since don't do tiling for argmin/argmax, we assume the init value
    // is always empty
    auto outValue = op.getValue();
    auto valueTy = cast<RankedTensorType>(outValue.getType());
    auto fpValueTy =
        RankedTensorType::get(valueTy.getShape(), rewriter.getF32Type());
    Value valueEmpty = rewriter.create<tensor::EmptyOp>(loc, valueTy.getShape(),
                                                        rewriter.getF32Type());

    auto outIdx = op.getIndex();
    auto axis = op.getAxis();
    auto newOp =
        rewriter.create<MKOpT>(loc, TypeRange{fpValueTy, outIdx.getType()},
                               fpInput, valueEmpty, outIdx, axis);
    Value fpValue = createElemwiseNaryOp<arith::FPToSIOp>(
        rewriter, loc, ValueRange{newOp.getResults()[0]}, outValue);
    rewriter.replaceOp(op, ValueRange{fpValue, newOp.getResults()[1]});
    return success();
  }
};

struct BoolOpShapeCanonicalizePattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool linearizeShape(linalg::GenericOp op, PatternRewriter &rewriter) const {
    assert(op.getOutputs().size() == 1 && "Only support single output");
    assert(llvm::all_of(op.getIndexingMapsArray(),
                        [](AffineMap &map) { return map.isIdentity(); }) &&
           "All affine maps must be identity affine map.");

    Location loc = op->getLoc();
    auto dstTensorType = cast<RankedTensorType>(op.getOutputs()[0].getType());

    if (dstTensorType.getRank() == 1)
      return false;

    auto elemCount = dstTensorType.getNumElements();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value elemCountVal = rewriter.create<arith::ConstantIntOp>(
        loc, elemCount, rewriter.getI32Type());

    auto indices = llvm::seq<int64_t>(0, dstTensorType.getRank());
    SmallVector<ReassociationIndices> reassociation(1);
    reassociation[0].insert(reassociation[0].end(), indices.begin(),
                            indices.end());

    SmallVector<Value> inputs1D = llvm::map_to_vector(
        llvm::concat<Value>(op.getInputs(), op.getOutputs()),
        [&](Value val) -> Value {
          return rewriter.create<tensor::CollapseShapeOp>(loc, val,
                                                          reassociation);
        });

    Value output1D = inputs1D.pop_back_val();
    SmallVector<AffineMap> idMaps(inputs1D.size() + 1,
                                  rewriter.getMultiDimIdentityMap(1));
    SmallVector<mlir::utils::IteratorType> iters(
        1, mlir::utils::IteratorType::parallel);
    auto newOp = rewriter.create<linalg::GenericOp>(
        loc, RankedTensorType::get({elemCount}, dstTensorType.getElementType()),
        inputs1D, ValueRange{output1D}, idMaps, iters);
    newOp.getRegion().takeBody(op.getRegion());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, dstTensorType, newOp->getResult(0), reassociation);

    return true;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1)
      return failure();

    Location loc = op->getLoc();
    auto elemWiseOp = regionOps[0];
    OperationName OpName = elemWiseOp->getName();
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();

    // Check if the operation is a boolean operation
    if (!(isa<arith::CmpFOp, math::IsNaNOp>(elemWiseOp)))
      return failure();

    if (linearizeShape(op, rewriter))
      return success();

    auto inputTensorType = cast<RankedTensorType>(inputs[0].getType());
    auto dstTensorType = cast<RankedTensorType>(outputs[0].getType());
    auto elemCount = dstTensorType.getNumElements();

    assert(dstTensorType.getRank() == 1);

    if (!(elemCount & 0x7))
      return failure();

    Value result = rewriter.create<tensor::EmptyOp>(
        loc, dstTensorType.getShape(), dstTensorType.getElementType());
    // Legalize operations that are not multiples of 8
    unsigned mainCount = elemCount & ~0x7;
    if (mainCount) {

      SmallVector<Value> ins = llvm::map_to_vector(
          llvm::concat<Value>(inputs, outputs), [&](Value val) -> Value {
            return rewriter.create<tensor::ExtractSliceOp>(
                loc,
                RankedTensorType::get(
                    {mainCount},
                    cast<RankedTensorType>(val.getType()).getElementType()),
                val, ValueRange(), ValueRange(), ValueRange(),
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{mainCount},
                ArrayRef<int64_t>{1});
          });

      Value out = ins.pop_back_val();
      SmallVector<AffineMap> idMaps(inputs.size() + 1,
                                    rewriter.getMultiDimIdentityMap(1));
      SmallVector<mlir::utils::IteratorType> iters(
          1, mlir::utils::IteratorType::parallel);
      auto newOp = rewriter.create<linalg::GenericOp>(
          loc,
          RankedTensorType::get({mainCount}, dstTensorType.getElementType()),
          ins, ValueRange{out}, idMaps, iters);
      newOp.getRegion().takeBody(op.getRegion());

      result = rewriter.create<tensor::InsertSliceOp>(
          loc, newOp.getResult(0), result, ValueRange(), ValueRange(),
          ValueRange(), ArrayRef<int64_t>{0}, ArrayRef<int64_t>{mainCount},
          ArrayRef<int64_t>{1});
    }

    for (unsigned idx = mainCount; idx < elemCount; ++idx) {
      auto idxVal = rewriter.create<arith::ConstantIndexOp>(loc, idx);
      auto loadIns = llvm::map_to_vector(inputs, [&](Value source) {
        return rewriter.create<tensor::ExtractOp>(loc, source,
                                                  ValueRange{idxVal});
      });
      IRMapping mapper;
      mapper.map(elemWiseOp->getOperands(), loadIns);
      auto newVal = rewriter.clone(*elemWiseOp, mapper);

      result = rewriter.create<tensor::InsertOp>(loc, newVal->getResult(0),
                                                 result, ValueRange{idxVal});
    }

    rewriter.replaceOp(op, result);

    return success();
  }

  BoolOpShapeCanonicalizePattern(MLIRContext *context, bool precisionPriority)
      : OpRewritePattern<linalg::GenericOp>(context),
        precisionPriority(precisionPriority) {}

private:
  bool precisionPriority = false;
};

struct SigmoidFusionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool matchSigmoid(linalg::GenericOp op, Value &input) const {
    // 1. sub (0 - x = -x)
    // 2. exp (e^(-x))
    // 3. add (1 + e^(-x))
    // 4. div (1 / (1 + e(^-x)))
    // We match the sigmoid pattern from down to up.

    // 1. Match div first.
    if (!checkGenericOp<arith::DivFOp>(op)) {
      return false;
    }

    auto divLhs = op.getInputs()[0];
    if (!isOneTensor(divLhs)) {
      return false;
    }

    // 2. Match add.
    auto addResult = op.getInputs()[1];
    auto addGenericOp = addResult.getDefiningOp<linalg::GenericOp>();
    if (!addGenericOp || !checkGenericOp<arith::AddFOp>(addGenericOp)) {
      return false;
    }

    auto addLhs = addGenericOp.getInputs()[0];
    auto addRhs = addGenericOp.getInputs()[1];
    bool isAddLhsOne = isOneTensor(addLhs);
    bool isAddRhsOne = isOneTensor(addRhs);
    if (!isAddLhsOne && !isAddRhsOne) {
      return false;
    }

    // 3. Match exp.
    auto expResult = isAddLhsOne ? addRhs : addLhs;
    auto expGenericOp = expResult.getDefiningOp<linalg::GenericOp>();
    if (!expGenericOp || !checkGenericOp<math::ExpOp>(expGenericOp)) {
      return false;
    }

    // 4. Match sub.
    auto subResult = expGenericOp.getInputs()[0];
    auto subGenericOp = subResult.getDefiningOp<linalg::GenericOp>();
    if (!subGenericOp || !checkGenericOp<arith::SubFOp>(subGenericOp)) {
      return false;
    }

    auto subLhs = subGenericOp.getInputs()[0];
    if (!isZeroTensor(subLhs)) {
      return false;
    }

    // Set input of Sub operation to the input of the sigmoid op.
    input = subGenericOp.getInputs()[1];

    // Match sigmoid pattern successfully.
    return true;
  }

public:
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Match sigmoid pattern
    Location loc = op.getLoc();
    Value input;
    if (!matchSigmoid(op, input)) {
      return rewriter.notifyMatchFailure(op, "sigmoid pattern not matched");
    }

    auto dstType = cast<RankedTensorType>(op.getType(0));
    auto elementType = dstType.getElementType();
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);

    // Replace the div GenericOp with mk::SigmoidOp
    // We can use CSE to erase other unused generic ops.
    auto sigmoidOp = rewriter.replaceOpWithNewOp<mk::SigmoidOp>(
        op, dstType, input, init, rewriter.getBoolAttr(false));

    return success();
  }
};

struct GeluFusionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isErfScaleTensor(Value &v) const {
    const float kSqrtScaleOverPiF32 = 1.0f / std::sqrt(2.0f);
    const float kSqrtScaleOverPiF16 = 0.70703125f;
    const float kSqrtScaleOverPiBF16 = 0.70703125f;
    auto elementType = cast<RankedTensorType>(v.getType()).getElementType();
    float erfScale;
    if (elementType.isF32()) {
      erfScale = kSqrtScaleOverPiF32;
    } else if (elementType.isF16()) {
      erfScale = kSqrtScaleOverPiF16;
    } else if (elementType.isBF16()) {
      erfScale = kSqrtScaleOverPiBF16;
    } else {
      return false;
    }
    return isConstantTensor(v, erfScale);
  }

  bool matchGeluErf(linalg::GenericOp op, Value input) const {
    // Match the mul op: (x * scale)
    auto mulErfResult = op.getInputs()[0];
    auto mulErfGenericOp = mulErfResult.getDefiningOp<linalg::GenericOp>();
    if (!mulErfGenericOp || !checkGenericOp<arith::MulFOp>(mulErfGenericOp)) {
      return false;
    }
    auto mulErfLhs = mulErfGenericOp.getInputs()[0];
    auto mulErfRhs = mulErfGenericOp.getInputs()[1];
    if (!isErfScaleTensor(mulErfRhs) && !isErfScaleTensor(mulErfLhs)) {
      return false;
    }

    Value input1 = isErfScaleTensor(mulErfRhs) ? mulErfLhs : mulErfRhs;

    return input1 == input;
  }

  bool isTanhScaledTensor(Value &v) const {
    // Check if the value is a constant tensor with the value of sqrt(2 / pi).
    const float kSqrt2OverPiF32 = std::sqrt(2.0f / M_PI);
    const float kSqrt2OverPiF16 = 0.7978515625f;
    const float kSqrt2OverPiBF16 = 0.796875f;
    auto elementType = cast<RankedTensorType>(v.getType()).getElementType();
    float tanhScale;
    if (elementType.isF32()) {
      tanhScale = kSqrt2OverPiF32;
    } else if (elementType.isF16()) {
      tanhScale = kSqrt2OverPiF16;
    } else if (elementType.isBF16()) {
      tanhScale = kSqrt2OverPiBF16;
    } else {
      return false; // Unsupported element type
    }
    return isConstantTensor(v, tanhScale);
  }

  bool isPowScaledTensor(Value &v) const {
    // Check if the value is a constant tensor with the value of 0.044715.
    const float powScale = 0.044715f;
    return isConstantTensor(v, powScale, true);
  }

  bool isAddAndMulOp(linalg::GenericOp op1, linalg::GenericOp op2) const {
    // Check if the given linalg generic op is an add and mul op.
    return (checkGenericOp<arith::AddFOp>(op1) &&
            checkGenericOp<arith::MulFOp>(op2)) ||
           (checkGenericOp<arith::AddFOp>(op2) &&
            checkGenericOp<arith::MulFOp>(op1));
  }

  bool isExtfAndAddOp(linalg::GenericOp op1, linalg::GenericOp op2) const {
    // Check if the given linalg generic op is an extf and add op.
    return (checkGenericOp<arith::ExtFOp>(op1) &&
            checkGenericOp<arith::AddFOp>(op2)) ||
           (checkGenericOp<arith::ExtFOp>(op2) &&
            checkGenericOp<arith::AddFOp>(op1));
  }

  // According LHS and RHS of the outer mul op, get the nested mul generic op
  // If the element type is F16 or BF16, we need to extend the input.
  // If the element type is F32, we can directly use the input.
  linalg::GenericOp getNestedMulGenericOp(linalg::GenericOp lhsGenericOp,
                                          linalg::GenericOp rhsGenericOp,
                                          bool isBit16) const {
    if (!isBit16) { // match case : mul (mul(lhs * rhs) * add (lhs1 *rhs1))
      if (!isAddAndMulOp(lhsGenericOp, rhsGenericOp)) {
        return linalg::GenericOp();
      }
      return checkGenericOp<arith::MulFOp>(lhsGenericOp) ? lhsGenericOp
                                                         : rhsGenericOp;
    } else { // match case : mul (extf (mul (lhs * rhs)) * add (lhs1 *rhs1))
      if (!isExtfAndAddOp(lhsGenericOp, rhsGenericOp)) {
        return linalg::GenericOp();
      }
      auto extfGenericOp = checkGenericOp<arith::ExtFOp>(lhsGenericOp)
                               ? lhsGenericOp
                               : rhsGenericOp;
      auto nestedMulGenericOp =
          extfGenericOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
      return (!nestedMulGenericOp ||
              !checkGenericOp<arith::MulFOp>(nestedMulGenericOp))
                 ? linalg::GenericOp()
                 : nestedMulGenericOp;
    }
  }

  bool matchGeluTanh(linalg::GenericOp op, Value input, bool isBit16) const {
    // 1. pow (pow(x, 2)))
    // 2. mul (0.044715 * pow(x, 2)))
    // 3. add (1 + 0.044715 * pow(x, 2)))
    // 4. mul (x * 0.79788456)
    // 5. mul (x * 0.79788456 * (1 + 0.044715 * pow(x, 2)))

    // We match the gelu tanh pattern from down to up.

    // 1. Match the mul op
    auto mulOfTanhResult = op.getInputs()[0];
    auto mulOfTanhGenericOp =
        mulOfTanhResult.getDefiningOp<linalg::GenericOp>();
    if (!mulOfTanhGenericOp ||
        !checkGenericOp<arith::MulFOp>(mulOfTanhGenericOp)) {
      return false;
    }

    auto mulOfTanhLhs = mulOfTanhGenericOp.getInputs()[0];
    auto mulOfTanhRhs = mulOfTanhGenericOp.getInputs()[1];
    auto mulOfTanhLhsGenericOp =
        mulOfTanhLhs.getDefiningOp<linalg::GenericOp>();
    auto mulOfTanhRhsGenericOp =
        mulOfTanhRhs.getDefiningOp<linalg::GenericOp>();
    if (!mulOfTanhLhsGenericOp || !mulOfTanhRhsGenericOp) {
      return false;
    }

    // Get the nested mul generic op.
    // If the element type is F16 or BF16, match the extf op and add op :
    // mul_result = mul (extf (mul (x, 0.79788456)), add (1 , operand))
    // If the element type is F32, match the mul op and add op :
    // mul_result = mul (mul (x, 0.79788456), add (1 , operand))
    linalg::GenericOp nestedMulGenericOp = getNestedMulGenericOp(
        mulOfTanhLhsGenericOp, mulOfTanhRhsGenericOp, isBit16);
    if (!nestedMulGenericOp) {
      return false;
    }

    // 2. Match the mul op: mul (x * 0.79788456)
    auto nestedMulInput1 = nestedMulGenericOp.getInputs()[0];
    auto nestedMulInput2 = nestedMulGenericOp.getInputs()[1];

    if (!isTanhScaledTensor(nestedMulInput1) &&
        !isTanhScaledTensor(nestedMulInput2)) {
      // If both operands are not scale tensors, we cannot match the gelu
      // pattern.
      return false;
    }
    Value input1 =
        isTanhScaledTensor(nestedMulInput1) ? nestedMulInput2 : nestedMulInput1;
    if (input1 != input) {
      // If the inputs of the mul ops are not the same, we cannot match the gelu
      // pattern.
      return false;
    }

    // 3. Match add (1 + 0.044715 * pow(x, 2)))
    auto nestedAddGenericOp =
        checkGenericOp<arith::AddFOp>(mulOfTanhLhsGenericOp)
            ? mulOfTanhLhsGenericOp
            : mulOfTanhRhsGenericOp;
    auto nestedAddLhs = nestedAddGenericOp.getInputs()[0];
    auto nestedAddRhs = nestedAddGenericOp.getInputs()[1];
    if (!isOneTensor(nestedAddLhs) && !isOneTensor(nestedAddRhs)) {
      return false;
    }

    // 4. Match the mul op: mul (0.044715 * pow(x, 2)))
    auto finalMulResult =
        isOneTensor(nestedAddLhs) ? nestedAddRhs : nestedAddLhs;
    auto finalMulGenericOp = finalMulResult.getDefiningOp<linalg::GenericOp>();
    if (!finalMulGenericOp ||
        !checkGenericOp<arith::MulFOp>(finalMulGenericOp)) {
      return false;
    }
    auto finalMulLhs = finalMulGenericOp.getInputs()[0];
    auto finalMulRhs = finalMulGenericOp.getInputs()[1];
    if (!isPowScaledTensor(finalMulLhs) && !isPowScaledTensor(finalMulRhs)) {
      // If both operands are not half tensors, we cannot match the gelu
      // pattern.
      return false;
    }

    // 5. Match pow (pow(x, 2)))
    auto powResult = isPowScaledTensor(finalMulRhs) ? finalMulLhs : finalMulRhs;
    auto powGenericOp = powResult.getDefiningOp<linalg::GenericOp>();
    if (!powGenericOp || !checkGenericOp<math::FPowIOp>(powGenericOp)) {
      return false;
    }

    auto powLhs = powGenericOp.getInputs()[0];
    auto powRhs = powGenericOp.getInputs()[1];
    if (!isTwoTensor(powRhs)) {
      // If the exponent is not a two tensor, we cannot match the gelu pattern.
      return false;
    }

    if (!isBit16) {
      return powLhs == input;
    }
    // If the element type is F16 or BF16, match the extf op nested pow op :
    // pow_result = pow (extf (x), 2)
    auto finalExtfGenericOp = powLhs.getDefiningOp<linalg::GenericOp>();
    if (!finalExtfGenericOp ||
        !checkGenericOp<arith::ExtFOp>(finalExtfGenericOp)) {
      return false;
    }

    return finalExtfGenericOp.getInputs()[0] == input;
  }

  bool matchGelu(linalg::GenericOp op, Value &input, GeluMode &geluMode) const {
    // Match gelu none or gelu tanh pattern.
    // match 0.5 * x * (1 + tanh/erf)

    // 1. match mul first.
    bool isBit16 = false;
    linalg::GenericOp mulGenericOp = op;
    if (checkGenericOp<arith::TruncFOp>(op) &&
        dyn_cast<RankedTensorType>(op.getType(0)).getElementTypeBitWidth() ==
            16) {
      isBit16 = true;
      mulGenericOp = op.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    }

    if (!mulGenericOp || !checkGenericOp<arith::MulFOp>(mulGenericOp)) {
      return false;
    }

    auto mulLhs = mulGenericOp.getInputs()[0];
    auto mulRhs = mulGenericOp.getInputs()[1];
    auto mulLhsGenericOp = mulLhs.getDefiningOp<linalg::GenericOp>();
    auto mulRhsGenericOp = mulRhs.getDefiningOp<linalg::GenericOp>();
    if (!mulLhsGenericOp || !mulRhsGenericOp) {
      return false;
    }

    // Get the nested mul generic op.
    // If the element type is F16 or BF16, and is tanh op:
    // mul_result = trunf( mul (extf (mul (x, 0.5)), add (1, tanh)) )
    // If the element type is F32, match the mul op and add op :
    // mul_result = mul (mul (x, 0.5), add (1, tanh/erf))
    linalg::GenericOp nestedMulOp =
        getNestedMulGenericOp(mulLhsGenericOp, mulRhsGenericOp, isBit16);
    if (!nestedMulOp) {
      return false;
    }

    // 2. Match the mul op: mul (0.5 * x)
    auto nestedMulLhs = nestedMulOp.getInputs()[0];
    auto nestedMulRhs = nestedMulOp.getInputs()[1];
    bool isNestedMulLhsHalf = isHalfTensor(nestedMulLhs);
    bool isNestedMulRhsHalf = isHalfTensor(nestedMulRhs);
    if (!isNestedMulLhsHalf && !isNestedMulRhsHalf) {
      // If both operands are not half tensors, we cannot match the gelu
      // pattern.
      return false;
    }
    Value input1 = isNestedMulRhsHalf ? nestedMulLhs : nestedMulRhs;

    // 3. Match add (1 + tanh/erf).
    auto addGenericOp = checkGenericOp<arith::AddFOp>(mulLhsGenericOp)
                            ? mulLhsGenericOp
                            : mulRhsGenericOp;
    auto addLhs = addGenericOp.getInputs()[0];
    auto addRhs = addGenericOp.getInputs()[1];
    bool isAddLhsOne = isOneTensor(addLhs);
    bool isAddRhsOne = isOneTensor(addRhs);
    if (!isAddLhsOne && !isAddRhsOne) {
      return false;
    }

    // 4. Match tanh/erf.
    auto tanhOrErfResult = isAddLhsOne ? addRhs : addLhs;
    auto tanhOrErfGenericOp =
        tanhOrErfResult.getDefiningOp<linalg::GenericOp>();
    if (!tanhOrErfGenericOp) {
      return false;
    }
    if (checkGenericOp<math::TanhOp>(tanhOrErfGenericOp) &&
        matchGeluTanh(tanhOrErfGenericOp, input1, isBit16)) {
      geluMode = GeluMode::Tanh;
      input = input1;
      return true;
    }
    if (checkGenericOp<math::ErfOp>(tanhOrErfGenericOp) &&
        matchGeluErf(tanhOrErfGenericOp, input1)) {
      geluMode = GeluMode::None;
      input = input1;
      return true;
    }

    // If the tanh/erf op is not matched, we cannot match the gelu pattern.
    return false;
  }

public:
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Match gelu pattern
    Location loc = op.getLoc();
    Value input;
    GeluMode geluMode = GeluMode::None;
    if (!matchGelu(op, input, geluMode)) {
      // If the gelu pattern is not matched, we cannot rewrite the op.
      return rewriter.notifyMatchFailure(op, "gelu pattern not matched");
    }

    auto dstType = cast<RankedTensorType>(op.getType(0));
    auto elementType = dstType.getElementType();
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);

    // Replace the mul generic op with mk::GeluOp
    switch (geluMode) {
    case GeluMode::None: {
      rewriter.replaceOpWithNewOp<mk::GeluOp>(
          op, dstType, input, nullptr, init, rewriter.getBoolAttr(false),
          rewriter.getI16IntegerAttr(static_cast<uint16_t>(geluMode)));
      break;
    }
    case GeluMode::Tanh: {
      // Double elementcount F32 buffer for immediate variable
      auto imm = rewriter.create<tensor::EmptyOp>(
          loc, dstType.getNumElements() * 2, rewriter.getF32Type());
      rewriter.replaceOpWithNewOp<mk::GeluOp>(
          op, dstType, input, imm, init, rewriter.getBoolAttr(false),
          rewriter.getI16IntegerAttr(static_cast<uint16_t>(geluMode)));
      break;
    }
    default: {
      llvm::report_fatal_error("Unsupported gelu mode!");
    }
    }

    return success();
  }
};

// copy from newest llvm lib
mlir::tensor::CollapseShapeOp
dropGivenUnitDims(OpBuilder &b, Location loc, Value src,
                  const llvm::SmallBitVector &dropDims) {
  auto srcType = cast<ShapedType>(src.getType());
  int64_t rank = srcType.getRank();
  assert(rank == static_cast<int64_t>(dropDims.size()) &&
         "dropDims dimension does not match src tensor rank");
  assert(llvm::all_of(
             dropDims.set_bits(),
             [&](unsigned dim) { return srcType.getShape()[dim] == 1; }) &&
         "Dropping non unit dimension");
  // Computed reassociation map for the corresponding tensor.collapse_shape.
  SmallVector<ReassociationIndices, 2> reassocMaps;
  // Current reassociation group to add dropped dimension to.

  int64_t nextDimToGroup = 0;
  llvm::SmallBitVector keptDims(dropDims);
  keptDims.flip();
  int64_t lastSetBit = keptDims.find_last();
  for (int64_t setBit : keptDims.set_bits()) {
    // Group consecutive dropped dimension with the next non-dropped dimension.
    // If this is the last set dimension, also group all subsequent dropped
    // dimension, if any.
    int64_t upTo = setBit == lastSetBit ? rank - 1 : setBit;
    auto seq = llvm::seq_inclusive(nextDimToGroup, upTo);
    reassocMaps.emplace_back(llvm::make_range(seq.begin(), seq.end()));
    nextDimToGroup = setBit + 1;
  }
  return b.create<tensor::CollapseShapeOp>(loc, src, reassocMaps);
}

template <typename MKOpT>
struct ArgMinMaxFusionPattern : OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  bool checkReductionBaseAttr(Value outVal) const {
    if (outVal.getDefiningOp<tensor::EmptyOp>())
      return true;
    if (auto fillOp = outVal.getDefiningOp<linalg::FillOp>()) {
      // TODO: check init is argmin/argmax identity value.
      return fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>();
    }
    assert(false && "Unsupported init op");
    return false;
  }

public:
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBody()->getNumArguments() != 4 ||
        op.getDimensions().size() != 1) {
      return failure();
    }

    // Get input and output types
    auto input = op.getInputs()[0];
    auto outVal = op.getInits()[0];
    auto outIdx = op.getInits()[1];

    if (!checkReductionBaseAttr(outVal))
      return rewriter.notifyMatchFailure(
          op.getLoc(), "mk.argmin/max not support non-identity init\n");

    auto inputType = cast<TensorType>(input.getType());
    auto valueType = cast<TensorType>(outVal.getType());
    auto indexType = cast<TensorType>(outIdx.getType());
    auto inputShape = inputType.getShape();

    // skip unsupport dtype
    auto elementType = inputType.getElementType();
    auto bitWidth = elementType.getIntOrFloatBitWidth();
    if (bitWidth == 64 && elementType.isF64()) {
      return failure();
    }

    assert(bitWidth == 16 || bitWidth == 32 || bitWidth == 64);

    // Get the reduction block and its operations
    auto block = op.getBody();
    auto ops = block->without_terminator();

    // Extract block arguments for current and reduced values/indices
    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    // Match the ArgMin/ArgMax pattern in the block
    bool isArgMin = std::is_same<MKOpT, mk::ArgMinOp>::value;
    auto opsIter = ops.begin();
    Value indexSelectOp, valueSelectOp;
    if (failed(matchArgMinMax(currValue, currIndex, reduceValue, reduceIndex,
                              opsIter, indexSelectOp, valueSelectOp,
                              isArgMin))) {
      return failure();
    }

    // Verify the terminator operation matches expected pattern
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIter << "\n");
    auto termOp = dyn_cast<linalg::YieldOp>(*opsIter++);
    if (!termOp || termOp != block->getTerminator())
      return failure();
    if (termOp.getOperands() != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
      return failure();
    }

    auto loc = op->getLoc();
    auto reduceDim = op.getDimensions()[0];
    int64_t reduceSize = inputShape[reduceDim];
    bool keepDim = inputType.getRank() == valueType.getRank();
    assert(valueType.getRank() == indexType.getRank());
    // Support reduce on last dim only temporary
    // TODO: for other dim, we can create a new contiguous buffer and copy in
    assert(reduceDim == inputType.getRank() - 1);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> lbs, ubs, steps;
    for (auto [i, size] : enumerate(inputShape)) {
      if (i != reduceDim) {
        auto sizeValue = rewriter.create<arith::ConstantIndexOp>(loc, size);
        lbs.push_back(zero);
        ubs.push_back(sizeValue);
        steps.push_back(one);
      }
    }

    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange{outVal, outIdx},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange indices,
            ValueRange iterArgs) {
          SmallVector<OpFoldResult> inputOffsets, inputSizes, inputStrides;
          SmallVector<OpFoldResult> outputOffsets, outputSizes, outputStrides;
          llvm::SmallBitVector dropDims;
          for (auto [i, size] : enumerate(inputShape)) {
            if (i == reduceDim) {
              inputOffsets.push_back(rewriter.getIndexAttr(0));
              inputSizes.push_back(rewriter.getIndexAttr(size));
              dropDims.push_back(false);
            } else {
              inputOffsets.push_back(indices[i < reduceDim ? i : i - 1]);
              inputSizes.push_back(rewriter.getIndexAttr(1));
              dropDims.push_back(true);
            }
            inputStrides.push_back(rewriter.getIndexAttr(1));
          }
          for (auto [i, size] : enumerate(valueType.getShape())) {
            if (keepDim && i == reduceDim)
              outputOffsets.push_back(rewriter.getIndexAttr(0));
            else
              outputOffsets.push_back(indices[i < reduceDim ? i : i - keepDim]);
            outputSizes.push_back(rewriter.getIndexAttr(1));
            outputStrides.push_back(rewriter.getIndexAttr(1));
          }
          auto inputVec = nestedBuilder.create<tensor::ExtractSliceOp>(
              loc, input, inputOffsets, inputSizes, inputStrides);
          auto inputCollapsed =
              dropGivenUnitDims(rewriter, loc, inputVec, dropDims);
          auto outValVec = nestedBuilder.create<tensor::ExtractSliceOp>(
              loc, iterArgs[0], outputOffsets, outputSizes, outputStrides);
          auto outIdxVec = nestedBuilder.create<tensor::ExtractSliceOp>(
              loc, iterArgs[1], outputOffsets, outputSizes, outputStrides);
          auto argOp = nestedBuilder.create<MKOpT>(
              loc, TypeRange{outValVec.getType(), outIdxVec.getType()},
              inputCollapsed, outValVec, outIdxVec, 0);
          auto outValTensor = nestedBuilder.create<tensor::InsertSliceOp>(
              loc, argOp.getResult()[0], iterArgs[0], outputOffsets,
              outputSizes, outputStrides);
          auto outIdxTensor = nestedBuilder.create<tensor::InsertSliceOp>(
              loc, argOp.getResult()[1], iterArgs[1], outputOffsets,
              outputSizes, outputStrides);
          return SmallVector<Value>{outValTensor, outIdxTensor};
        });
    rewriter.replaceAllUsesWith(op->getResults(), loopNest.results);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReduceOpToElementwiseOpConverter
    : public ReduceScanOpConversionBase<linalg::ReduceOp, linalg::YieldOp> {
private:
  using ReduceScanOpConversionBase::ReduceScanOpConversionBase;

  // memref<?xi1>: Assume base is least 8 bit align. offset is calculated as
  // byte. So we don't expected extract i1 in bytes.
  SmallVector<Value> lowerBool1DInput(ConversionPatternRewriter &rewriter,
                                      Location loc, Type elementType,
                                      ValueRange inputs,
                                      linalg::ReduceOp op) const {

    auto rop = getRegionOps<linalg::ReduceOp>(op).front();

    auto attr = getRedBaseAttr(rewriter, rop, elementType);

    auto finalResult =
        createReduceOp(rewriter, op, loc, inputs, SmallVector<int64_t>{0},
                       SmallVector<int64_t>{}, elementType, attr)
            .getResults();
    return finalResult;
  }

  SmallVector<Value>
  lower1DInput(ValueRange inputs, linalg::ReduceOp op,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto shape = inputType.getShape();

    int32_t tileSize = shape[0] > 64 ? 64 : shape[0];

    SmallVector<Value> lastRes(inputs.size());
    assert(inputs.size() == 1 && "Expected only one input for 1D reduction");

    // NOTE: Use scf.while may exist dynamic shape problem.
    // shape > 64: tiling n * 64, reduction n dim
    // remain 64: tiling 2 * n, reduction half parts
    if (shape[0] > 64) {
      // Reshape to 2D tensor with shape [tile, N / tile]
      // Call lowering leading dimension reduction
      SmallVector<int64_t> tiledShape = {shape[0] >> 6, 64};
      Value reshape = rewriter.create<tensor::ExpandShapeOp>(
          loc, RankedTensorType::get(tiledShape, inputType.getElementType()),
          inputs[0], ArrayRef<ReassociationIndices>{{0, 1}});
      lastRes = lowerLeadingDimension({reshape}, op, rewriter);
    } else {
      lastRes = inputs;
    }

    if (inputType.getElementType().isInteger(1)) {
      // TODO: Can optimized to only 8 elements
      return lowerBool1DInput(rewriter, loc, inputType.getElementType(),
                              lastRes, op);
    }

    Region &combineOp = op.getRegion();
    auto createExtractSliceOp = [&](Value val,
                                    SmallVector<int64_t> static_offsets,
                                    SmallVector<int64_t> static_size,
                                    SmallVector<int64_t> static_stride) {
      return rewriter.create<tensor::ExtractSliceOp>(
          loc, RankedTensorType::get(static_size, inputType.getElementType()),
          val, ValueRange(), /*sizes*/ ValueRange(),
          /*strides*/ ValueRange(), static_offsets, static_size, static_stride);
    };

    for (int32_t i = tileSize >> 1; i >= 1; i >>= 1) {
      auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);

      auto curRes = createExtractSliceOp(
          lastRes.front(), SmallVector<int64_t>{0}, SmallVector<int64_t>{i},
          SmallVector<int64_t>{1});
      auto RHS = createExtractSliceOp(lastRes.front(), SmallVector<int64_t>{i},
                                      SmallVector<int64_t>{i},
                                      SmallVector<int64_t>{1});
      lastRes = accumulate({RHS}, {curRes}, combineOp, rewriter);
    }
    // Collapse the shape of the last result to a scalar tensor
    std::transform(
        lastRes.begin(), lastRes.end(), lastRes.begin(), [&](auto val) {
          return rewriter.create<tensor::CollapseShapeOp>(
              loc, RankedTensorType::get({}, inputType.getElementType()), val,
              ArrayRef<ReassociationIndices>{});
        });

    return lastRes;
  }

  SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, linalg::ReduceOp op,
                        ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto shape = inputType.getShape();

    // Initialize accumulators as empty tensors of shape [shape[1], ..]
    SmallVector<int64_t> accShape(shape.begin() + 1, shape.end());

    auto results = op.getResults();
    SmallVector<Value> acc(results.size());

    // Build offsets, sizes, and strides for ExtractSlice
    SmallVector<int64_t> static_offsets(shape.size(), 0);

    SmallVector<int64_t> sizeVal({1});
    sizeVal.insert(sizeVal.end(), shape.begin() + 1, shape.end());

    SmallVector<int64_t> strides(shape.size(), 1);
    // {1,shape[axis+1],..shape[rank]} ->
    // {shape[axis+1],..shape[rank]}
    SmallVector<ReassociationIndices> reassociation(shape.size() - 1);
    // The first group: [0, 1]
    reassociation[0].resize(2);
    std::iota(reassociation[0].begin(), reassociation[0].end(), 0);
    // The remaining groups: [2, ..., shape.size()-1]
    for (size_t i = 2; i < shape.size(); ++i) {
      reassociation[i - 1].push_back(i);
    }

    std::transform(inputs.begin(), inputs.end(), acc.begin(), [&](auto val) {
      auto extract_tensor = rewriter.create<tensor::ExtractSliceOp>(
          loc, RankedTensorType::get(sizeVal, inputType.getElementType()), val,
          /*offset*/ ValueRange(), /*sizes*/ ValueRange(),
          /*strides*/ ValueRange(),
          /*static_offsets*/
          static_offsets, sizeVal, strides);

      // {1,shape[axis+1],..shape[rank]} ->
      // {shape[axis+1],..shape[rank]}
      return rewriter.create<tensor::CollapseShapeOp>(loc, extract_tensor,
                                                      reassociation);
    });

    // scf.for loop bounds and step
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, acc,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          SmallVector<Value> currAcc = iterArgs;

          // Build offsets, sizes, and strides for ExtractSlice
          SmallVector<Value> dynOffsets = {iv};
          for (size_t j = 1; j < shape.size(); ++j) {
            dynOffsets.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
          }

          // iv is a Value, so build dynamic offsets for ExtractSlice
          SmallVector<Value> subInputs(inputs.size());

          std::transform(
              inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
                auto extract_tensor = b.create<tensor::ExtractSliceOp>(
                    loc,
                    RankedTensorType::get(sizeVal, inputType.getElementType()),
                    val, dynOffsets, /*sizes*/ ValueRange(),
                    /*strides*/ ValueRange(),
                    /*static_offsets*/
                    SmallVector<int64_t>(shape.size(), ShapedType::kDynamic),
                    sizeVal, strides);

                // {1,shape[axis+1],..shape[rank]} ->
                // {shape[axis+1],..shape[rank]}
                return rewriter.create<tensor::CollapseShapeOp>(
                    loc, extract_tensor, reassociation);
              });
          currAcc = accumulate(subInputs, currAcc, combineOp, b);
          b.create<scf::YieldOp>(loc, currAcc);
        });

    // Extract result tensors from forOp
    return forOp.getResults();
  }

  uint32_t getAxis(linalg::ReduceOp op) const override {
    // For linalg.reduce, the axis is always 0.
    auto dims = op.getDimensions();
    assert(dims.size() == 1 && "Expected a single dimension");
    return dims[0];
  }

  SmallVector<Value> getInputs(linalg::ReduceOp op) const override {
    // For linalg.reduce, we return the inputs directly.
    return op.getInputs();
  }
};

struct ArithRemFRewrite : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteRemFOp(linalg::GenericOp op,
                              PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto input1 = op.getInputs()[0];
    auto input2 = op.getInputs()[1];

    auto divResult = buildLinalgElementwise<arith::DivFOp>(
        rewriter, loc, ValueRange{input1, input2});
    auto truncResult = buildLinalgElementwise<math::TruncOp>(
        rewriter, loc, ValueRange{divResult});
    auto mulResult = buildLinalgElementwise<arith::MulFOp>(
        rewriter, loc, ValueRange{truncResult, input2});
    auto subResult = buildLinalgElementwise<arith::SubFOp>(
        rewriter, loc, ValueRange{input1, mulResult});

    rewriter.replaceOp(op, subResult);
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1) {
      return failure();
    }

    auto bodyOp = regionOps[0];
    if (!isa<arith::RemFOp>(bodyOp))
      return failure();

    return rewriteRemFOp(op, rewriter);
  }
};

struct I1ExtUIOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    auto regionOps = triton::getRegionOps<linalg::GenericOp>(op);

    if (regionOps.size() != 1 || !isa<arith::ExtUIOp>(regionOps.front()))
      return rewriter.notifyMatchFailure(op, "only rewrite i1 extension op\n");

    auto extOp = cast<arith::ExtUIOp>(regionOps.front());

    if (!extOp->getOperandTypes()[0].isInteger(1))
      return rewriter.notifyMatchFailure(op, "only rewrite i1 extension op\n");

    Location loc = op.getLoc();

    auto input = op.getInputs()[0];
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);

    Type f16Type = rewriter.getF16Type();
    auto empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), f16Type);

    auto castType = RankedTensorType::get(inputType.getShape(), f16Type);
    auto f16Reseult =
        rewriter.create<mk::Bit2FpOp>(loc, castType, input, empty)
            ->getResult(0);

    auto rank = resultType.getRank();
    SmallVector<AffineMap, 2> indexingMaps(
        2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    auto dstEmpty = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto result = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{f16Reseult},
        ValueRange{dstEmpty}, indexingMaps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto src = args[0];
          auto fPToSIOp = builder.create<arith::FPToSIOp>(
              loc, resultType.getElementType(), src);
          builder.create<linalg::YieldOp>(loc, ValueRange{fPToSIOp});
        });

    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

struct I1ExtSIOpRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    auto regionOps = triton::getRegionOps<linalg::GenericOp>(op);

    if (regionOps.size() != 1 || !isa<arith::ExtSIOp>(regionOps.front()))
      return rewriter.notifyMatchFailure(op, "only rewrite i1 extension op\n");

    auto extOp = cast<arith::ExtSIOp>(regionOps.front());

    if (!extOp->getOperandTypes()[0].isInteger(1))
      return rewriter.notifyMatchFailure(op, "only rewrite i1 extension op\n");

    Location loc = op.getLoc();

    auto input = op.getInputs()[0];
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op->getResultTypes()[0]);

    Type f16Type = rewriter.getF16Type();
    auto empty =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), f16Type);

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, f16Type, rewriter.getZeroAttr(f16Type));
    Value negOne = rewriter.create<arith::ConstantOp>(
        loc, f16Type, rewriter.getFloatAttr(f16Type, -1.0));

    auto zeroTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
            .result();
    auto negOneTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{negOne}, ValueRange{empty})
            .result();

    auto castType = RankedTensorType::get(inputType.getShape(), f16Type);
    auto maskCast = rewriter.create<mk::Bit2FpOp>(loc, castType, input, empty);
    Value maskmoveResult =
        rewriter
            .create<mk::MaskMoveOp>(loc, negOneTensor.getType(), negOneTensor,
                                    maskCast->getResult(0), zeroTensor)
            ->getResult(0);

    auto rank = resultType.getRank();
    SmallVector<AffineMap, 2> indexingMaps(
        2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    auto dstEmpty = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto result = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{maskmoveResult},
        ValueRange{dstEmpty}, indexingMaps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto src = args[0];
          auto fPToSIOp = builder.create<arith::FPToSIOp>(
              loc, resultType.getElementType(), src);
          builder.create<linalg::YieldOp>(loc, ValueRange{fPToSIOp});
        });

    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};

struct AssertOpConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!assertToCf) {
      return convertToMK(op, adaptor, rewriter);
    } else {
      return convertToCF(op, adaptor, rewriter);
    }
  }

private:
  static LogicalResult convertToCF(triton::AssertOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) {
    Value condVal = op.getCondition();

    if (isa<mlir::TensorType>(condVal.getType())) {
      auto scalarVal = getScalarValue(op.getCondition(), op.getLoc(), rewriter);
      condVal = scalarVal ? scalarVal : condVal;
    }
    assert(condVal && isa<mlir::IntegerType>(condVal.getType()) &&
           "Only asserts on scalars are currently supported");

    if (!condVal.getType().isInteger(1)) {
      auto zero =
          rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 32);
      auto newCond = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, condVal, zero);
      condVal = newCond.getResult();
    }

    auto assertMessage =
        llvm::formatv("Assertion `{0}` failed", op.getMessage());
    rewriter.create<mlir::cf::AssertOp>(op.getLoc(), condVal,
                                        assertMessage.str());

    rewriter.eraseOp(op);
    return success();
  }

  static LogicalResult convertToMK(triton::AssertOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) {
    auto loc = op->getLoc();
    auto condition = adaptor.getCondition();
    auto conditionType = condition.getType();
    Value isTrueCondition;

    if (conditionType.isInteger(1)) {
      isTrueCondition = condition;
    } else if (auto tensorType = cast<TensorType>(conditionType)) {
      if (!tensorType.getElementType().isInteger(1)) {
        return rewriter.notifyMatchFailure(
            op, "Condition tensor must have i1 element type");
      }
      auto trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      auto emptyInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{}, tensorType.getElementType());
      auto filledInit = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{trueVal},
                                                    ValueRange{emptyInit})
                            .getResult(0);
      int64_t rank = tensorType.getRank();
      SmallVector<int64_t> dimensions;
      for (int64_t i = 0; i < rank; ++i) {
        dimensions.push_back(i);
      }

      auto reduceOp = rewriter.create<linalg::ReduceOp>(
          loc, ValueRange{condition}, ValueRange{filledInit}, dimensions,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value inputElem = args[0];
            Value accumulated = args[1];
            auto anyFalse =
                b.create<arith::AndIOp>(loc, accumulated, inputElem);
            b.create<linalg::YieldOp>(loc, anyFalse->getResult(0));
          });
      isTrueCondition =
          rewriter.create<tensor::ExtractOp>(loc, reduceOp.getResult(0));
    } else {
      return rewriter.notifyMatchFailure(
          op, "Condition must be i1 scalar or tensor with i1 element type");
    }
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, isTrueCondition,
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<mk::AssertOp>(loc, TypeRange{}, adaptor.getMessage());
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }

  bool assertToCf = false;
};
} // namespace

void mlir::triton::populateLinalgToMKPreProcessPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<NormalizeReduceInitToIdentityPattern, // Need before si-to-fp.
                SigmoidFusionPattern,
                GeluFusionPattern,
                ArgMinMaxFusionPattern<mk::ArgMinOp>,
                ArgMinMaxFusionPattern<mk::ArgMaxOp>,
                AtomicRMWOpRewrite,
                AtomicCASOpRewrite>(
      patterns.getContext());
  // clang-format on
}

void mlir::triton::populateLinalgToMKTypeConversionPatterns(
    RewritePatternSet &patterns, bool precisionPriority) {
  patterns.add<CastElementwiseOpIOToFloatPattern, CastReduceOpIOToFloatPattern>(
      patterns.getContext(), precisionPriority /* precisionPriority */);
  patterns.add<CannonicalizeRedudantTypeConversion>(patterns.getContext());
  patterns.add<I1ExtSIOpRewrite, I1ExtUIOpRewrite>(patterns.getContext());
  // TODO: if need precision mode
  patterns.add<CastArgMinMaxOpIOToFloatPattern<mk::ArgMaxOp>,
               CastArgMinMaxOpIOToFloatPattern<mk::ArgMinOp>>(
      patterns.getContext());
}

void mlir::triton::populateLinalgToMKCanonicalizationPatterns(
    RewritePatternSet &patterns, bool precisionPriority) {
  // clang-format off
  patterns.add<LinalgReduceToMKReduceConversion, // Exec after NormalizeReduceInitToIdentityPattern and si-to-fp
                BroadcastOpRewrite,
                SelectOpRewrite,
                MinMaxOpRewrite,
                IsInfOpRewrite,
                MKDotScaleOpRewrite,
                LinalgMatmulOpRewrite,
                ScalarGlobalLoadRewrite<affine::AffineLoadOp>,
                ScalarGlobalLoadRewrite<memref::LoadOp>,
                ScalarGlobalStoreRewrite<affine::AffineStoreOp>,
                ScalarGlobalStoreRewrite<memref::StoreOp>>(
      patterns.getContext());
  // clang-format on

  if (!precisionPriority)
    patterns.add<ArithRemFRewrite, DivFloatOpRewrite, PowFOpRewrite>(
        patterns.getContext());
  else
    patterns.add<DivIntOpRewrite>(patterns.getContext());
}

void mlir::triton::populateLinalgToMKShapeCanonicalizationPatterns(
    RewritePatternSet &patterns, bool precisionPriority) {
  patterns.add<BoolOpShapeCanonicalizePattern>(
      patterns.getContext(), precisionPriority /* precisionPriority */);
}

void mlir::triton::populateLinalgToMKConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AssertOpConverter>(patterns.getContext());
  // After NormalizeReduceInitToIdentityPattern and si-to-fp
  patterns.add<ReduceOpToElementwiseOpConverter>(patterns.getContext());
}