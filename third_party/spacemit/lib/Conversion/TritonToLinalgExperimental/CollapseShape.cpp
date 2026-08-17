//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "collapse-shape"

using namespace mlir;
using namespace triton;

namespace {

// This pattern collapses a `linalg.fill` operation that fills a tensor with a
// single value into a `tensor.expand_shape` operation that expands a tensor
// filled with the same value to a larger shape. This is useful for optimizing
// the performance of tensor operations that involve broadcasting or filling
// large tensors with a single value.
// //
// for example:
// linalg.fill
// before
// ```
//     %13 = linalg.fill ins(%c1_i32 : i32) outs(%12 :
//     tensor<1x1x1x1x1x2x1xi32>) -> tensor<1x1x1x1x1x2x1xi32>
// ```
// after
// ```
//     %17 = tensor.collapse_shape %12 ...
//     %18 = linalg.fill ins(%c1_i32 : i32) outs(%17 : tensor<2xi32>) ->
//     tensor<2xi32> %expanded_6 = tensor.expand_shape %18 [[0, 1, 2, 3, 4, 5,
//     6]] output_shape [1, 1, 1, 1, 1, 2, 1] : tensor<2xi32> into
//     tensor<1x1x1x1x1x2x1xi32>
// ```
struct CollapseFill : public OpRewritePattern<linalg::FillOp> {
  CollapseFill(MLIRContext *context)
      : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult collapseMemRef(linalg::FillOp op,
                               PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto result = op.getOutputs()[0];
    auto resultType = mlir::dyn_cast_or_null<MemRefType>(result.getType());
    if (!resultType)
      return failure();
    auto rank = resultType.getRank();
    if (rank <= 1 ||
        dyn_cast_or_null<StridedLayoutAttr>(resultType.getLayout())) {
      return failure();
    }
    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.push_back({});
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(i));
    }
    auto elementType = resultType.getElementType();

    auto output = memref::CollapseShapeOp::create(
        rewriter, loc,
        MemRefType::get(llvm::ArrayRef<int64_t>{resultType.getNumElements()},
                        elementType),
        result, reassociationMap);
    op.getOutputsMutable()[0].set(output.getResult());
    return success();
  }

  LogicalResult collapseTensor(linalg::FillOp op,
                               PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto result = op.getResult(0);
    auto resultType =
        mlir::dyn_cast_or_null<RankedTensorType>(result.getType());
    if (!resultType)
      return failure();
    auto rank = resultType.getRank();
    if (rank <= 1) {
      return failure();
    }
    auto elementType = resultType.getElementType();

    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.push_back({});
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(i));
    }

    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({resultType.getNumElements()}, elementType),
        op.getOutputs()[0], reassociationMap);
    auto fillOp =
        linalg::FillOp::create(rewriter, loc, op.getInputs(), ValueRange{init});

    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, loc, result.getType(), fillOp.getResult(0), reassociationMap);

    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() <= 1 && "code assumes single result!");
    if (op->getNumResults() == 1) {
      return collapseTensor(op, rewriter);
    } else if (op->getNumResults() == 0) {
      return collapseMemRef(op, rewriter);
    }
    return failure();
  }
};

// This pattern collapses a `linalg.transpose` operation that transposes a
// tensor into a `tensor.collapse_shape` operation that collapses the tensor
// to a smaller shape. This is useful for optimizing the performance of tensor
// operations that involve transposing large tensors.
// //
// for example:
// linalg.transpose
// before
// ```
//     %transposed = linalg.transpose ins(%expanded_2 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) outs(%66 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) permutation = [0, 1, 2, 3, 4, 5, 6, 7,
//     9, 8]
// ```
// after
// ```
//     %collapsed = tensor.collapse_shape %expanded_17 [[0, 1, 2, 3, 4, 5, 6,
//     7], [8], [9]] : tensor<2x2x2x2x2x2x2x2x2x2xi64> into tensor<256x2x2xi64>
//     %77 = tensor.collapse_shape %66 ...:
//     %transposed = linalg.transpose ins(%collapsed : tensor<256x2x2xi64>)
//     outs(%77 : tensor<256x2x2xi64>) permutation = [0, 2, 1] %expanded_22 =
//     tensor.expand_shape  %transposed ...
// ```

struct CollapseTranspose : public OpRewritePattern<linalg::TransposeOp> {
  CollapseTranspose(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getInput();
    auto sourceType = dyn_cast_or_null<RankedTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    auto elementType = sourceType.getElementType();
    if (sourceRank <= 3) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 3 for transpose collapse");
    }

    SmallVector<int64_t> perm(op.getPermutation());
    SmallVector<int64_t> transposedShape(sourceRank);
    SmallVector<ReassociationExprs> reassociationMap;
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    int dim = 0;
    SmallVector<int64_t> permIdx(sourceRank);
    for (size_t i = 0; i < sourceRank; ++i) {
      permIdx[perm[i]] = i;
    }
    // The original dim corresponds to the dim after collapse
    SmallVector<int64_t> mapDim;
    for (size_t i = 0; i < sourceRank; ++i) {
      auto id = permIdx[i];
      if (i > 0 && (id == 0 || !(perm[id] == perm[id - 1] + 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      mapDim.push_back(dim);
      int64_t dimSize = sourceType.getDimSize(i);
      if (ShapedType::isDynamic(collapseShapeInput[dim]) ||
          ShapedType::isDynamic(dimSize)) {
        collapseShapeInput[dim] = ShapedType::kDynamic;
      } else {
        collapseShapeInput[dim] *= dimSize;
      }
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    SmallVector<int64_t> newPerm;
    for (size_t i = 0; i < sourceRank; ++i) {
      if (i > 0 && newPerm.back() == mapDim[perm[i]]) {
        continue;
      }
      newPerm.push_back(mapDim[perm[i]]);
    }
    perm = newPerm;
    // update transposedShape, based on perm
    transposedShape.clear();
    for (size_t i = 0; i < perm.size(); ++i) {
      transposedShape.push_back(collapseShapeInput[perm[i]]);
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    source = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, source,
                                             reassociationMap);

    // Build reassociationMapRe for collapsing op.getInit() into transposedShape
    // op.getInit() has shape [source[perm[0]], source[perm[1]], ...,
    // source[perm[n-1]]] We need to map this to transposedShape
    SmallVector<ReassociationExprs> reassociationMapRe(reassociationMap.size());
    int idx = 0;
    for (size_t i = 0; i < perm.size(); ++i) {
      int collapsedDim = mapDim[perm[i]];
      reassociationMapRe[collapsedDim].push_back(
          rewriter.getAffineDimExpr(idx++));
    }

    Value transposeInit = tensor::CollapseShapeOp::create(
        rewriter, loc, RankedTensorType::get(transposedShape, elementType),
        op.getInit(), reassociationMapRe);

    Value transpose =
        linalg::TransposeOp::create(rewriter, loc, source, transposeInit, perm)
            .getResults()[0];

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], transpose, reassociationMapRe);
    return success();
  }
};

// This pattern collapses a `linalg.generic` operation that broadcasts a tensor
// to a larger shape into a `tensor.expand_shape` operation that expands the
// tensor to the desired shape. This is useful for optimizing the performance
// //
// for example:
// linalg.generic with broadcast
// before
// ```
//     %79 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types =
//     ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel",
//     "parallel", "parallel", "parallel", "parallel"]} ins(%76 :
//     tensor<1x1x1x1x1x1x1x1x2x2xi32>) outs(%77 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi32>) attrs =  {broadcastDims = array<i64: 0,
//     1, 2, 3, 4, 5, 6, 7>} { ^bb0(%in: i32, %out: i32):
//       linalg.yield %in : i32
//     } -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
// after
// ```
//     %collapsed_30 = tensor.collapse_shape %89 [[0, 1, 2, 3, 4, 5, 6, 7], [8,
//     9]] : tensor<1x1x1x1x1x1x1x1x2x2xi32> into tensor<1x4xi32> %92 =
//     tensor.collapse_shape %77 [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]  :
//     tensor<2x2x2x2x2x2x2x2x2x2xi32> into tensor<256x4xi32> %93 =
//     linalg.generic {indexing_maps = [#map2, #map1], iterator_types =
//     ["parallel", "parallel"]} ins(%collapsed_30 : tensor<1x4xi32>) outs(%92 :
//     tensor<256x4xi32>) attrs =  {broadcastDims = array<i64: 0>} { ^bb0(%in:
//     i32, %out: i32):
//       linalg.yield %in : i32
//     } -> tensor<256x4xi32>
//     %expanded_31 = tensor.expand_shape %93 [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]
//     output_shape [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] : tensor<256x4xi32> into
//     tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
struct CollapseBroadCast : public OpRewritePattern<linalg::GenericOp> {
  CollapseBroadCast(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("broadcastDims")) {
      return rewriter.notifyMatchFailure(op,
                                         "expected broadcastDims attribute");
    }
    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto input = op.getInputs()[0];
    auto sourceType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    if (sourceRank <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 1 for broadcast collapse");
    }

    auto resultType =
        dyn_cast_or_null<RankedTensorType>(op.getResultTypes()[0]);
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for result");
    auto elementType = resultType.getElementType();
    // collapse input tensor from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t, 8> collapseShapeInput;
    SmallVector<int64_t, 8> collapseShapeOutput;
    SmallVector<ReassociationExprs, 8> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < sourceRank; i++) {
      if (i > 0 && !((sourceType.getDimSize(i) == 1 &&
                      sourceType.getDimSize(i - 1) == 1) ||
                     (sourceType.getDimSize(i) != 1 &&
                      sourceType.getDimSize(i - 1) != 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        collapseShapeOutput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
      collapseShapeOutput[dim] *= resultType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    input = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, input,
                                            reassociationMap);
    resultType = RankedTensorType::get(collapseShapeOutput, elementType);
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());
    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");

    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(resultType.getShape(), elementType),
        op.getOutputs()[0], reassociationMap);

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, init->getResultTypes(), ValueRange{input},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank));
    rewriter.cloneRegionBefore(op.getRegion(), linalgOp.getRegion(),
                               linalgOp.getRegion().begin());
    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], linalgOp->getResult(0), reassociationMap);
    return success();
  }
};

// This pattern collapses a `linalg.reduce` operation that reduces a tensor
// to a smaller shape by summing over specified dimensions into a
// `tensor.expand_shape` operation that expands the reduced tensor to the
// desired shape. This is useful for optimizing the performance of tensor
// operations that involve reductions.
// //
// for example:
// linalg.reduce
// before
// ```
//     %reduced = linalg.reduce ins(%transposed :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) outs(%68 :
//     tensor<2x2x2x2x2x2x2x2x2xi64>) dimensions = [8]
//       (%in: i64, %init: i64) {
//         %311 = arith.xori %in, %init : i64
//         linalg.yield %311 : i64
//       }
// ```
// after
// ```
//     %collapsed_20 = tensor.collapse_shape %expanded_19 [[0, 1, 2, 3, 4, 5, 6,
//     7], [8]] : tensor<2x2x2x2x2x2x2x2x2xi64> into tensor<256x2xi64> %reduced
//     = linalg.reduce ins(%transposed : tensor<256x2x2xi64>) outs(%collapsed_20
//     : tensor<256x2xi64>) dimensions = [1]
//       (%in: i64, %init: i64) {
//         %377 = arith.xori %in, %init : i64
//         linalg.yield %377 : i64
//       }
//     %expanded_21 = tensor.expand_shape %reduced [[0, 1, 2, 3, 4, 5, 6, 7],
//     [8]] output_shape [2, 2, 2, 2, 2, 2, 2, 2, 2] : tensor<256x2xi64> into
//     tensor<2x2x2x2x2x2x2x2x2xi64>
// ```
struct CollapseReduce : public OpRewritePattern<linalg::ReduceOp> {
  CollapseReduce(MLIRContext *context)
      : OpRewritePattern<linalg::ReduceOp>(context) {}
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInputs()[0];
    auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for input");
    }
    auto inputRank = inputType.getRank();
    auto dims = op.getDimensions();
    if (inputRank - dims.size() <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected input rank - reduction loops > 1 for reduce collapse");
    }
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    SmallVector<int64_t> newDims;
    SmallVector<ReassociationExprs> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < inputRank; i++) {
      bool reduceAxis = llvm::is_contained(dims, i);
      if (i > 0 && (reduceAxis || llvm::is_contained(dims, i - 1))) {
        // reduce axis
        dim++;
      }
      if (reduceAxis) {
        newDims.push_back(dim);
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim] *= inputType.getDimSize(i);
    }
    if (collapseShapeInput.size() == inputRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse reduce shape");
    }
    SmallVector<int64_t> collapseShapeOutput;
    for (size_t i = 0; i < collapseShapeInput.size(); i++) {
      if (llvm::is_contained(newDims, i)) {
        // reduce axis
        continue;
      }
      collapseShapeOutput.push_back(collapseShapeInput[i]);
    }
    auto elementType = inputType.getElementType();
    auto loc = op.getLoc();
    auto newInputType = RankedTensorType::get(collapseShapeInput, elementType);
    input = tensor::CollapseShapeOp::create(rewriter, loc, newInputType, input,
                                            reassociationMap);

    SmallVector<ReassociationExprs> reassociationMapOutput;
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      if (llvm::is_contained(newDims, i)) {
        continue; // skip reduce axis
      } else {
        reassociationMapOutput.push_back({});
      }
      for (size_t j = 0; j < reassociationMap[i].size(); ++j) {
        reassociationMapOutput.back().push_back(
            rewriter.getAffineDimExpr(idx++));
      }
    }
    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc, RankedTensorType::get(collapseShapeOutput, elementType),
        op.getInits()[0], reassociationMapOutput);
    auto newReduce =
        linalg::ReduceOp::create(rewriter, loc, init->getResultTypes(),
                                 ValueRange{input}, ValueRange{init}, newDims);
    rewriter.cloneRegionBefore(op.getRegion(), newReduce.getRegion(),
                               newReduce.getRegion().begin());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], newReduce->getResult(0),
        reassociationMapOutput);
    return success();
  }
};

class CollapseShapePasss : public OperationPass<ModuleOp> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollapseShapePasss)

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapseFill, CollapseBroadCast, CollapseTranspose,
                 CollapseReduce>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCollapseShapePass() {
  return std::make_unique<CollapseShapePasss>();
}
