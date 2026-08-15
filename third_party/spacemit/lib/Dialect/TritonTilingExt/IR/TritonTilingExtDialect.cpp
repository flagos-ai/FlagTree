//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttx;
using namespace mlir::linalg;

namespace mlir {
namespace ttx {

Value getSlice(OpBuilder &b, Location loc, Value source,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return tensor::ExtractSliceOp::create(b, loc, source, offsets, sizes,
                                              strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return memref::SubViewOp::create(b, loc, source, offsets, sizes,
                                         strides);
      })
      .Default([&](Type t) { return nullptr; });
}

//
// getTiledImplementation
//
// Given an array of offsets and sizes, return the corresponding tiled version
// of the current op.
//
// This method is responsible for creating the extract slice ops for each
// operand of the op (including input and output operand).
//
// As an example, assuming we tile a linalg.matmul ins(%0, %1) out(%out)
//
// This method then generate:
//
// %in_slice_0 = extract_slice from %0
// %in_slice_1 = extract_slice from %1
// %out_slice = extract_slice from %out
// %tile = linalg.matmul ins(%in_slice_0, %in_slice_1) out(%out_slice)
//
// To generate these extract slice, we go over each operand, get the
// corresponding affine map to compute the correct offsets and sizes.
//
// Now let's describe how we compute the correct offsets and sizes from
// an affine map.
//
// - Offsets:
// An affine map describes how to access a tensor (i.e: the indicies into a
// tensor), so getting the offsets (also indices) from an affine map is just
// simply "applying" the sub-map on the offset (calling
// makeComposedFoldedAffineApply which also does constant folding
// automatically).
//
// For example:
// Let's assume we have the following nested loops:
// for i in range(0, 10):
//   for j in range(0, 20):
//      for k in range(0, 30):
//         dst[i][j][k] = src[i * 2][j + k]
//
// Assume that we describe the iteration space based on dst. So:
// - dst's affine map is (d0, d1, d2) -> (d0, d1, d2)
// - src's affine map is (d0, d1, d2) -> (d0 * 2, d1 + d2)
//
// Now let's say we want to tile the operator with offset (0, 1, 2).
//
// For dst, we apply this (0, 1, 2) to its affine map and get (0, 1, 2)
//
// For src, we have to plug in the offsets into the affine map to get:
//
// (0 * 2, 1 + 2) = (0, 3)
//
// This is exactly what the implementation does as well.
// The call to getSubMap gets the i'th result expression, then the call to
// makeComposedFoldedAffineApply apply the `offsets` array to the i'th result
// expression in the affine map.
//
//
// - Sizes:
// Size is slightly more complex, notice that there are 3 steps to compute
// sizes:
//
// 1) call linalg::computeTileSizes on the provided `sizes`
// 2) apply the affine map
// 3) add 1 to the result
//
// The reason for this complexity is because the affine maps describe indices
// iteration space with a half open interval (i.e.: we always from 0 until right
// before the upper bound). So if we simply apply the affine map on the sizes,
// we will have incorrect results.
//
// Consider this snippet again:
//  for i in range(0, 16):
//    for j in range(0, 32):
//      for k in range(0, 64):
//         dst[i][j][k] = src[i * 2][j + k]
//
// Assume we want the operator to have a tile size of (16, 32, 64) -- so no
// tiling at all. If we apply the affine map of src (d0, d1, d2) -> (d0 * 2, d1
// + d2), we have
//
// (16 * 2, 32 + 64) -> (32, 96)
//
// However, consider the second dimension of source:
//   - j goes from 0 till 31 inclusive
//   - k goes from 0 till 63 inclusive
//
// So the max index of src's second dimension is 31 + 63 = 94. Since index
// starts from 0, this means the second dimension has 95 elements. But the
// formula gives us a tile size of 96!!! The same argument can be applied for
// the first dimension as well, the number of elements is 15 * 2 + 1 = 31, but
// computed tile size is 32.
//
// So simply applying the indexing map to compute tile size is INCORRECT!!
// This happens because the indexing map operates on [0, size), while tile sizes
// are inclusive.
//
// The correct formula is:
// ((d0 - 1) * 2 + 1), (d1 - 1) + (d2 - 1) + 1 which gives
// (15 * 2 + 1, 32 - 1 + 64 - 1 + 1) -> (31, 95)
//
// So again, the steps are:
// - Subtract 1 from the sizes (what linalg::computeTileSizes does)
// - Apply the affine map
// - Add 1 to the result
//
template <typename TritonTilingExtOpTy>
FailureOr<TilingResult> getTiledImplementation(TritonTilingExtOpTy op,
                                               OpBuilder &b,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes) {
  Location loc = op->getLoc();
  SmallVector<Value> valuesToTile = op->getOperands();
  SmallVector<Value> tiledValues;
  auto oneAttr = b.getI64IntegerAttr(1);

  for (OpOperand &opOperand : op->getOpOperands()) {
    unsigned int index = opOperand.getOperandNumber();
    auto val = valuesToTile[index];
    auto type = dyn_cast<ShapedType>(val.getType());

    if (!type) {
      tiledValues.push_back(val);
      continue;
    }

    auto rank = type.getRank();
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides(rank, oneAttr);

    llvm::SmallVector<mlir::OpFoldResult> composedTileSizes =
        linalg::computeTileSizes(b, loc, sizes, {});

    AffineMap map = op.getIndexingMap(b.getContext(), index, sizes);
    for (int64_t i = 0; i < rank; i++) {
      AffineMap m = map.getSubMap(i);
      {
        OpFoldResult upperboundClosed =
            affine::makeComposedFoldedAffineApply(b, loc, m, composedTileSizes);
        AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
        OpFoldResult size = affine::makeComposedFoldedAffineApply(
            b, loc, s0 + 1, upperboundClosed);
        newSizes.push_back(size);
      }
      {
        OpFoldResult offset =
            affine::makeComposedFoldedAffineApply(b, loc, m, offsets);
        newOffsets.push_back(offset);
      }
    }

    tiledValues.push_back(
        getSlice(b, loc, val, newOffsets, newSizes, newStrides));
  }

  SmallVector<Type> resultTensorTypes = llvm::to_vector(
      llvm::map_range(op.getDpsInitsMutable(), [&](OpOperand &opOperand) {
        return tiledValues[opOperand.getOperandNumber()].getType();
      }));

  Operation *tiledOp = clone(b, op, resultTensorTypes, tiledValues);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

//
// getResultTilePosition
// This method returns the resultOffsets and resultSizes through references
// for the tiled operator. While `getTiledImplementation` is responsible for
// generating the extract slice for all operands, `getResultTilePosition` is
// responsible for returning the offsets and sizes which the tiling engine will
// then use to generate the corresponding insert_slice ops.
//
// Because the slice we insert back to the output tensor is the same as the
// slice that we extracted from the output tensor, this method just repeats the
// offset and size computation in `getTiledImplementation`.
//
template <typename TritonTilingExtOpTy>
LogicalResult getResultTilePosition(TritonTilingExtOpTy op, OpBuilder &b,
                                    unsigned resultNumber,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVector<OpFoldResult> &resultOffsets,
                                    SmallVector<OpFoldResult> &resultSizes) {
  Location loc = op.getLoc();

  AffineMap outputMap =
      op.getOutputIndexingMap(b.getContext(), resultNumber, sizes);

  Value result = op.getDpsInitOperand(resultNumber)->get();
  auto rank = dyn_cast<ShapedType>(result.getType()).getRank();

  llvm::SmallVector<mlir::OpFoldResult> composedTileSizes =
      linalg::computeTileSizes(b, loc, sizes, {});
  for (int64_t i = 0; i < rank; i++) {
    AffineMap m = outputMap.getSubMap(i);
    {
      OpFoldResult upperboundClosed =
          affine::makeComposedFoldedAffineApply(b, loc, m, composedTileSizes);
      AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
      OpFoldResult size = affine::makeComposedFoldedAffineApply(
          b, loc, s0 + 1, upperboundClosed);
      resultSizes.push_back(size);
    }
    {
      OpFoldResult offset =
          affine::makeComposedFoldedAffineApply(b, loc, m, offsets);
      resultOffsets.push_back(offset);
    }
  }

  return success();
}

// This method is borrowed verbatim from
// mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp
//
// This is invoked when the current op produces a result that is used
// as an input to another op that is being tiled. The method essentially handles
// producing a new op where the result matches the given offsets and sizes.
// If the method succeeds, the two new operators will be fused in the same loop.
//
// As an example, consider the following IR where the linalg.generic is being
// tiled (unnecessary detailed omitted for brevity):
//
// clang-format: off
//
// func.func @some_op_1(
//     %arg0: tensor<8x2x256x512xbf16>,
//     %arg1: tensor<8x256x1024xbf16>
// ) -> tensor<8x256x1024xbf16>
//     %1 = linalg.init_tensor [8, 256, 1024] : tensor<8x256x1024xbf16>
//     %2 = linalg.init_tensor [8, 256, 1024] : tensor<8x256x1024xbf16>
//     %3 = ttx.some_op
//             ins(%arg0 : tensor<8x2x256x512xbf16>)
//             outs(%1 : tensor<8x256x1024xbf16>) -> tensor<8x256x1024xbf16>
//     %4 = linalg.generic
//         ins(%3, %arg1 : tensor<8x256x1024xbf16>, tensor<8x256x1024xbf16>)
//         outs(%2 : tensor<8x256x1024xbf16>) {
//     ^bb0(%arg2: bf16, %arg3: bf16, %arg4: bf16):
//       %add = arith.addf %arg2, %arg3 : bf16
//       linalg.yield %add : bf16
//     } -> tensor<8x256x1024xbf16>
//     return %4 : tensor<8x256x1024xbf16>
// }
//
// clang-format: on
//
// We tile linalg.generic, but one of its inputs is %3 which is the result of
// ttx.some_op. So the tiling engine will invoke
// generateResultTileValue of ttx.some_op to determine if it's
// possible to create a tiled version of it, thereby making it possible to fuse
// both operators together in a loop.
template <typename TritonTilingExtOpTy>
FailureOr<TilingResult>
generateResultTileValue(TritonTilingExtOpTy op, OpBuilder &b,
                        unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes) {

  // Check that the indexing map used for the output is a projected
  // permutation. This could be relaxed with a more general approach that can
  // map the offsets and sizes from the result to iteration space tiles
  // (filling in full extent for dimensions not used to access the result).
  AffineMap indexingMap = op.getOutputIndexingMap(b.getContext(), 0, sizes);
  if (!indexingMap.isProjectedPermutation()) {
    return op.emitOpError(
        "unhandled tiled implementation generation when result is not "
        "accessed using a permuted projection");
  }

  auto numLoops = op.getLoopIteratorTypes().size();
  SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
      iterationTileSizes(numLoops);
  if (!indexingMap.isPermutation()) {
    SmallVector<Range> iterationDomain = op.getIterationDomain(b);
    for (auto range : llvm::enumerate(iterationDomain)) {
      iterationTileOffsets[range.index()] = range.value().offset;
      iterationTileSizes[range.index()] = range.value().size;
    }
  }
  for (auto resultExpr : llvm::enumerate(indexingMap.getResults())) {
    assert(resultExpr.value().getKind() == AffineExprKind::DimId);
    // HACK: LLVM casting utilities do not work here for out-of-tree builds,
    // as there is no template specialization for this cast in the base
    // build.
    AffineDimExpr affineDimExpr(static_cast<AffineExpr::ImplType *>(
        const_cast<void *>(resultExpr.value().getAsOpaquePointer())));
    unsigned dimPosition = affineDimExpr.getPosition();
    iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
    iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
  }

  FailureOr<TilingResult> tilingResult =
      op.getTiledImplementation(b, iterationTileOffsets, iterationTileSizes);
  if (tilingResult->tiledOps.size() != 1)
    return op.emitOpError("failed to generate tiled implementation");

  return TilingResult{
      tilingResult->tiledOps,
      SmallVector<Value>{tilingResult->tiledValues[resultNumber]}};
}

// This method is borrowed directly from linalg.generic's implementation
// in mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
// This marks all operands that are part of the input group to have read
// effect, while all other operands that are part of the output group
// to have both read and write effects.
static void getTritonTilingExtEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ArrayRef<OpOperand *> inputOperands,
    const MutableOperandRange &outputOperands) {
  for (auto operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand->get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
  for (auto &operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;

    effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

template <typename TritonTilingExtOpTy>
void getEffects(
    TritonTilingExtOpTy op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getTritonTilingExtEffectsImpl(effects, op.getOperation()->getResults(),
                                op.getDpsInputOperands(),
                                op.getDpsInitsMutable());
}

} // namespace ttx
} // namespace mlir

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonTilingExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOps.cpp.inc"

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOpsDialect.cpp.inc"
