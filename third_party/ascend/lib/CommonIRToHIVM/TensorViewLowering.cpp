/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ascend/include/CommonIRToHIVM/TensorViewLowering.h"
#include "ascend/include/CommonIRToHIVM/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
namespace {
// ========== Helper ======== //
// Physical (flat) element offset and in-bounds predicate of one tile element.
// A struct rather than a pair because both fields are captured by the scf.if
// builder lambdas below, and capturing a structured binding is C++20.
struct ElementAccess {
  Value physOffset;
  Value inBounds;
};

// Dense tile origin. `index` must already have index type.
Value emitDenseOrigin(OpBuilder &b, Location loc, Value index,
                      int64_t traversal) {
  Value step = b.create<arith::ConstantIndexOp>(loc, traversal);
  return b.create<arith::MulIOp>(loc, index, step);
}

// Resolve one view dim at a required, index-typed tile coordinate.
Value emitLogicalCoord(OpBuilder &b, Location loc,
                       const TensorViewAccessInfo &info, ValueRange indices,
                       unsigned dim, Value coord) {
  if (llvm::is_contained(info.sparseDims, static_cast<int64_t>(dim))) {
    auto raw =
        b.create<tensor::ExtractOp>(loc, indices[dim], ValueRange{coord});
    raw->setAttr("DiscreteMemAccess", b.getUnitAttr());
    return asIndexValue(b, loc, raw.getResult());
  }
  Value index = asIndexValue(b, loc, indices[dim]);
  Value origin = emitDenseOrigin(b, loc, index, info.traversal[dim]);
  return b.create<arith::AddIOp>(loc, origin, coord);
}

// Add one view dim's contribution to the flat offset. `logical` has index
// type; a null `offset` starts the sum without emitting an addition to zero.
Value emitAccumulateFlatOffset(OpBuilder &b, Location loc,
                               const TensorViewAccessInfo &info, unsigned dim,
                               Value logical, Value offset) {
  unsigned baseDim = static_cast<unsigned>(info.dimMap[dim]);
  Value stride = asIndexValue(b, loc, info.strides[baseDim]);
  Value physical = b.create<arith::MulIOp>(loc, logical, stride);
  return offset ? b.create<arith::AddIOp>(loc, offset, physical) : physical;
}

// Resolve each coordinate, check bounds, and accumulate its flat offset. A
// sparse dim is checked only when the view enables bounds checking.
ElementAccess emitElementAccess(OpBuilder &b, Location loc,
                                const TensorViewAccessInfo &info,
                                ValueRange indices, ValueRange coords) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value inBounds = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
  Value physOffset;
  for (unsigned dim = 0; dim < info.rank(); ++dim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[dim]);
    Value logical = emitLogicalCoord(b, loc, info, indices, dim, coords[dim]);

    bool sparse =
        llvm::is_contained(info.sparseDims, static_cast<int64_t>(dim));
    if (!sparse || info.boundsCheck) {
      Value extent = asIndexValue(b, loc, info.shape[baseDim]);
      Value nonNeg = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                             logical, zero);
      Value below = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                            logical, extent);
      Value dimIn = b.create<arith::AndIOp>(loc, nonNeg, below);
      inBounds = b.create<arith::AndIOp>(loc, inBounds, dimIn);
    }

    physOffset =
        emitAccumulateFlatOffset(b, loc, info, dim, logical, physOffset);
  }
  return ElementAccess{physOffset, inBounds};
}

// ======== For PerIndexPayload ===========//
Value emitSparseInBoundsMask(OpBuilder &b, Location loc,
                             const TensorViewAccessInfo &info,
                             ValueRange indices, unsigned dim) {
  // Every operand below must be indices[dim], not indices[0]: each sparse dim
  // carries its own index tensor, and those tensors differ in length (one per
  // tile extent), so reading dim 0's tensor here would both test the wrong
  // coordinates and build a mask of the wrong shape.
  auto idxTy = cast<RankedTensorType>(indices[dim].getType());
  Type idxElem = idxTy.getElementType();

  unsigned baseDim = static_cast<unsigned>(info.dimMap[dim]);
  Value extent = asIndexValue(b, loc, info.shape[baseDim]);
  Value extentElem = b.create<arith::IndexCastOp>(loc, idxElem, extent);

  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(idxElem));
  Value zeroSplat = b.create<tensor::SplatOp>(loc, idxTy, zero);
  Value extentSplat = b.create<tensor::SplatOp>(loc, idxTy, extentElem);

  Value nonNeg = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                         indices[dim], zeroSplat);
  Value below = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                        indices[dim], extentSplat);
  return b.create<arith::AndIOp>(loc, nonNeg, below);
}

/// Resolve the logical coordinate of every dim before the payload, for one
/// iteration of the loop nest. A sparse dim reads its own index tensor at that
/// dim's induction variable; a dense dim offsets its tile origin by the same.
SmallVector<Value> emitLeadingLogical(OpBuilder &b, Location loc,
                                      const TensorViewAccessInfo &info,
                                      ValueRange indices,
                                      ArrayRef<Value> coords) {
  SmallVector<Value> leadingLogical;
  for (unsigned d = 0; d < coords.size(); ++d)
    leadingLogical.push_back(
        emitLogicalCoord(b, loc, info, indices, d, coords[d]));
  return leadingLogical;
}

/// Per-dim bounds masks for the leading segment, indexed by view dim and null
/// for a dense dim. Dense bounds are handled by clipping the loop upper bound;
/// sparse dims still need masks because valid indices need not form a prefix.
/// Without bounds checking no sparse mask is built and every entry is null.
SmallVector<Value> emitLeadingMasks(OpBuilder &b, Location loc,
                                    const TensorViewAccessInfo &info,
                                    ValueRange indices, unsigned nLeading) {
  SmallVector<Value> masks(nLeading, Value());
  if (!info.boundsCheck)
    return masks;
  for (unsigned d = 0; d < nLeading; ++d)
    if (llvm::is_contained(info.sparseDims, static_cast<int64_t>(d)))
      masks[d] = emitSparseInBoundsMask(b, loc, info, indices, d);
  return masks;
}

/// Valid prefix length of one dense GatherScatterView dim.
/// GatherScatterView has traversal == tile > 0, so a negative block index
/// makes the entire dimension invalid and yields zero length. Otherwise the
/// valid region is the prefix [0, max(0, min(extent - origin, tile))).
/// This depends only on the scalar block index and base extent, not loop
/// coords.
Value emitDenseValidLength(OpBuilder &b, Location loc,
                           const TensorViewAccessInfo &info, ValueRange indices,
                           unsigned dim) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  unsigned baseDim = static_cast<unsigned>(info.dimMap[dim]);
  Value index = asIndexValue(b, loc, indices[dim]);
  Value origin = emitDenseOrigin(b, loc, index, info.traversal[dim]);
  Value extent = asIndexValue(b, loc, info.shape[baseDim]);
  Value remaining = b.create<arith::SubIOp>(loc, extent, origin);
  Value tileSize = b.create<arith::ConstantIndexOp>(loc, info.tile[dim]);
  Value capped = b.create<arith::MinSIOp>(loc, remaining, tileSize);
  Value length = b.create<arith::MaxSIOp>(loc, capped, zero);
  Value nonNegative =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, zero);
  return b.create<arith::SelectOp>(loc, nonNegative, length, zero);
}

/// Per-view-dim sizes: dense dims are clipped to their valid prefix, while
/// sparse dims retain the full tile size and use separate bounds masks.
/// Leading sizes become loop upper bounds; payload sizes become DMA sizes.
SmallVector<Value> emitClippedTileSizes(OpBuilder &b, Location loc,
                                        const TensorViewAccessInfo &info,
                                        ValueRange indices) {
  SmallVector<Value> sizes;
  for (unsigned d = 0; d < info.rank(); ++d) {
    if (llvm::is_contained(info.sparseDims, static_cast<int64_t>(d)))
      sizes.push_back(b.create<arith::ConstantIndexOp>(loc, info.tile[d]));
    else
      sizes.push_back(emitDenseValidLength(b, loc, info, indices, d));
  }
  return sizes;
}

/// Flat element offset of one payload block.
///
/// `leadingLogical` holds the already-resolved logical coordinate of every dim
/// before the payload, in view-dim order. The caller resolves those inside the
/// loop nest, since a leading sparse dim reads its own index tensor while a
/// leading dense dim adds its induction variable to the tile origin. The
/// payload dims contribute only their dense tile origin; the actual transfer
/// lengths are determined separately by the clipped sizes.
Value emitPayloadBaseOffset(OpBuilder &b, Location loc,
                            const TensorViewAccessInfo &info,
                            ValueRange indices,
                            ArrayRef<Value> leadingLogical) {
  unsigned nLeading = leadingLogical.size();
  Value offset;

  for (unsigned dim = 0; dim < info.rank(); ++dim) {
    Value logical;
    if (dim < nLeading) {
      logical = asIndexValue(b, loc, leadingLogical[dim]);
    } else {
      Value index = asIndexValue(b, loc, indices[dim]);
      logical = emitDenseOrigin(b, loc, index, info.traversal[dim]);
    }

    offset = emitAccumulateFlatOffset(b, loc, info, dim, logical, offset);
  }

  return offset;
}

Value emitPayloadGmView(OpBuilder &b, Location loc,
                        const TensorViewAccessInfo &info, Value offset,
                        unsigned nLeading, ValueRange validLengths) {
  SmallVector<int64_t> shape(info.tile.begin(), info.tile.end());
  for (unsigned d = 0; d < nLeading; ++d)
    shape[d] = 1;
  // Keep a payload dim's extent static when its valid length folds to a
  // constant -- which it does whenever the base extent is a constexpr, i.e. the
  // common case. Marking it dynamic unconditionally would hide that from the
  // DMA lowering below.
  for (unsigned d = nLeading; d < info.rank(); ++d) {
    APInt folded;
    if (matchPattern(validLengths[d - nLeading], m_ConstantInt(&folded)))
      shape[d] = folded.getSExtValue();
    else
      shape[d] = ShapedType::kDynamic;
  }

  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> strideStatic;

  for (unsigned dim = 0; dim < info.tile.size(); ++dim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[dim]);
    if (dim < nLeading)
      sizes.push_back(b.getIndexAttr(1));
    else
      sizes.push_back(validLengths[dim - nLeading]);

    int64_t stride = info.staticStrides[baseDim];
    strideStatic.push_back(stride);
    if (stride == ShapedType::kDynamic)
      strides.push_back(info.strides[baseDim]);
    else
      strides.push_back(b.getIndexAttr(stride));
  }

  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       strideStatic);
  auto space =
      hivm::AddressSpaceAttr::get(b.getContext(), hivm::AddressSpace::GM);
  auto gmTy = MemRefType::get(shape, info.elementType, layout, space);

  return b.create<memref::ReinterpretCastOp>(
      loc, gmTy, info.base, OpFoldResult(offset), sizes, strides);
}

Value emitPayloadBufferSubview(OpBuilder &b, Location loc, Value buffer,
                               const TensorViewAccessInfo &info,
                               ArrayRef<Value> coords,
                               ValueRange validLengths) {
  unsigned nLeading = coords.size();
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  for (unsigned d = 0; d < info.rank(); ++d) {
    offsets.push_back(d < nLeading ? OpFoldResult(coords[d])
                                   : OpFoldResult(b.getIndexAttr(0)));
    sizes.push_back(d < nLeading ? OpFoldResult(b.getIndexAttr(1))
                                 : OpFoldResult(validLengths[d - nLeading]));
    strides.push_back(b.getIndexAttr(1));
  }

  return b.create<memref::SubViewOp>(loc, buffer, offsets, sizes, strides);
}

// ========== ScalarTransfer ======== //
Value emitScalarGather(OpBuilder &b, Location loc,
                       const TensorViewAccessInfo &info, ValueRange indices,
                       RankedTensorType resultTy, hivm::AddressSpace space,
                       Value pad) {
  Value buffer = b.create<memref::AllocOp>(
      loc, makeSpaceMemref(b.getContext(), info.tile, info.elementType, space));
  emitElementwiseFill(b, loc, info, indices, buffer, pad);
  return b.create<bufferization::ToTensorOp>(loc, resultTy, buffer, true,
                                             false);
}

LogicalResult emitScalarScatter(OpBuilder &b, Location loc,
                                const TensorViewAccessInfo &info,
                                ValueRange indices, Value source) {
  unsigned rank = info.rank();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  ElementAccess access = emitElementAccess(b, loc, info, indices, coords);
  b.create<scf::IfOp>(
      loc, access.inBounds, [&](OpBuilder &nested, Location nestedLoc) {
        Value element =
            nested.create<tensor::ExtractOp>(nestedLoc, source, coords);
        Value elemView = emitGmElementView(nested, nestedLoc, info.base,
                                           access.physOffset, info.elementType);
        // Keep tensor materialization for scalar GM writes: plain memref.store
        // into this element view has silently dropped lanes on the Ascend
        // backend.
        Value empty = nested.create<tensor::EmptyOp>(
            nestedLoc, ArrayRef<int64_t>{1}, info.elementType);
        Value inserted = nested.create<tensor::InsertOp>(
            nestedLoc, element, empty, ValueRange{zero});
        auto materialize =
            nested.create<bufferization::MaterializeInDestinationOp>(
                nestedLoc, inserted, elemView);
        materialize->setAttr("writable", nested.getUnitAttr());
        nested.create<scf::YieldOp>(nestedLoc);
      });
  return success();
}

// ========== PerIndexPayloadDMA ======== //
Value emitPerIndexPayloadGather(OpBuilder &b, Location loc,
                                const TensorViewAccessInfo &info,
                                const GatherScatterAccessInfo &access,
                                ValueRange indices, RankedTensorType type,
                                hivm::AddressSpace space, Value pad) {
  unsigned nLeading = access.payload.firstDenseDim;
  Value buffer = b.create<memref::AllocOp>(
      loc, makeSpaceMemref(b.getContext(), info.tile, info.elementType, space));
  // Pre-fill the whole tile; in-range rows overwrite their own slice below.
  b.create<hivm::VBrcOp>(loc, TypeRange{}, pad, buffer);

  SmallVector<Value> masks = emitLeadingMasks(b, loc, info, indices, nLeading);
  // Hoisted: clipped sizes do not depend on the loop coordinates.
  SmallVector<Value> tileSizes = emitClippedTileSizes(b, loc, info, indices);
  ArrayRef<Value> validLengths =
      ArrayRef<Value>(tileSizes).drop_front(nLeading);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  Value inBounds;
  // Scoped so the guard restores the insertion point before the to_tensor
  // below: an InsertionGuard only restores when it goes out of scope, so
  // building the result while still inside the nest would define it in a child
  // region and leave it not dominating its use outside.
  {
    OpBuilder::InsertionGuard guard(b);
    for (unsigned d = 0; d < nLeading; ++d) {
      auto loop = b.create<scf::ForOp>(loc, zero, tileSizes[d], one);
      b.setInsertionPointToStart(loop.getBody());
      coords.push_back(loop.getInductionVar());

      // Dense bounds are enforced by the loop upper bound. Fold each sparse
      // mask at its own level. inBounds stays null when bounds checking is off.
      if (masks[d]) {
        Value lane =
            b.create<tensor::ExtractOp>(loc, masks[d], ValueRange{coords[d]});
        inBounds =
            inBounds ? b.create<arith::AndIOp>(loc, inBounds, lane) : lane;
      }
    }

    auto loadBlock = [&](OpBuilder &nested, Location nestedLoc) {
      SmallVector<Value> leadingLogical =
          emitLeadingLogical(nested, nestedLoc, info, indices, coords);
      Value offset = emitPayloadBaseOffset(nested, nestedLoc, info, indices,
                                           leadingLogical);
      Value gm = emitPayloadGmView(nested, nestedLoc, info, offset, nLeading,
                                   validLengths);
      Value ubBlock = emitPayloadBufferSubview(nested, nestedLoc, buffer, info,
                                               coords, validLengths);

      nested.create<hivm::LoadOp>(nestedLoc, TypeRange{}, gm, ubBlock);
    };
    if (!inBounds) {
      loadBlock(b, loc);
    } else {
      b.create<scf::IfOp>(loc, inBounds,
                          [&](OpBuilder &nested, Location nestedLoc) {
                            loadBlock(nested, nestedLoc);
                            nested.create<scf::YieldOp>(nestedLoc);
                          });
    }
  }

  return b.create<bufferization::ToTensorOp>(loc, type, buffer, true, false);
}
/// Scatter one dense payload block per leading coordinate tuple, the mirror of
/// emitPerIndexPayloadGather. Each leading dim has one loop level: sparse dims
/// read their own index tensor, while dense dims add the loop coordinate to
/// their tile origin. A single payload DMA is guarded innermost.
///
/// The gather's pre-fill has no analogue here. A store cannot be redirected or
/// undone -- clamping an out-of-range coordinate would corrupt whatever block
/// it clamps onto. Dense loop bounds skip invalid coordinates; the scf.if drops
/// invalid sparse tuples. Sparse masks and loop bounds are computed before the
/// loop nest, and each mask lane is extracted at its own dim's loop level.
LogicalResult emitPerIndexPayloadScatter(OpBuilder &b, Location loc,
                                         const TensorViewAccessInfo &info,
                                         const GatherScatterAccessInfo &access,
                                         ValueRange indices, Value source,
                                         hivm::AddressSpace space) {
  unsigned nLeading = access.payload.firstDenseDim;
  if (nLeading == 0 || nLeading >= info.rank())
    return failure();

  // Stage the whole source tile in UB once; the per-block stores then take
  // subviews of it instead of re-materializing tensor slices per iteration.
  Value ub = b.create<memref::AllocOp>(
      loc, makeSpaceMemref(b.getContext(), info.tile, info.elementType, space));
  auto materialize =
      b.create<bufferization::MaterializeInDestinationOp>(loc, source, ub);
  materialize->setAttr("writable", b.getUnitAttr());

  SmallVector<Value> masks = emitLeadingMasks(b, loc, info, indices, nLeading);
  // Hoisted: clipped sizes do not depend on the loop coordinates.
  SmallVector<Value> tileSizes = emitClippedTileSizes(b, loc, info, indices);
  ArrayRef<Value> validLengths =
      ArrayRef<Value>(tileSizes).drop_front(nLeading);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  Value inBounds;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < nLeading; ++d) {
    auto loop = b.create<scf::ForOp>(loc, zero, tileSizes[d], one);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());

    // Dense bounds are enforced by the loop upper bound. Fold each sparse
    // mask at its own level. inBounds stays null when bounds checking is off.
    if (masks[d]) {
      Value lane =
          b.create<tensor::ExtractOp>(loc, masks[d], ValueRange{coords[d]});
      inBounds = inBounds ? b.create<arith::AndIOp>(loc, inBounds, lane) : lane;
    }
  }

  auto storeBlock = [&](OpBuilder &nested, Location nestedLoc) {
    SmallVector<Value> leadingLogical =
        emitLeadingLogical(nested, nestedLoc, info, indices, coords);
    Value offset =
        emitPayloadBaseOffset(nested, nestedLoc, info, indices, leadingLogical);
    Value gm = emitPayloadGmView(nested, nestedLoc, info, offset, nLeading,
                                 validLengths);
    Value ubBlock = emitPayloadBufferSubview(nested, nestedLoc, ub, info,
                                             coords, validLengths);

    nested.create<hivm::StoreOp>(nestedLoc, TypeRange{}, ubBlock, gm);
  };
  if (!inBounds) {
    storeBlock(b, loc);
    return success();
  }
  b.create<scf::IfOp>(loc, inBounds,
                      [&](OpBuilder &nested, Location nestedLoc) {
                        storeBlock(nested, nestedLoc);
                        nested.create<scf::YieldOp>(nestedLoc);
                      });

  return success();
}

// ========== SingleAffineDMA ======== //
FailureOr<Value> emitSingleAffineGather() { return failure(); }
LogicalResult emitSingleAffineScatter() { return failure(); }
} // namespace

void emitElementwiseFill(OpBuilder &b, Location loc,
                         const TensorViewAccessInfo &info, ValueRange indices,
                         Value buffer, Value pad) {
  unsigned rank = info.rank();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  ElementAccess access = emitElementAccess(b, loc, info, indices, coords);
  auto ifOp = b.create<scf::IfOp>(
      loc, access.inBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value elemView = emitGmElementView(nested, nestedLoc, info.base,
                                           access.physOffset, info.elementType);
        Value loaded = nested.create<memref::LoadOp>(nestedLoc, elemView,
                                                     ValueRange{zero});
        nested.create<scf::YieldOp>(nestedLoc, loaded);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        nested.create<scf::YieldOp>(nestedLoc, pad);
      });
  b.create<memref::StoreOp>(loc, ifOp.getResult(0), buffer, coords);
}

FailureOr<Value> emitGatherLoad(OpBuilder &b, Location loc,
                                const TensorViewAccessInfo &info,
                                ValueRange indices, RankedTensorType type,
                                hivm::AddressSpace space, Value pad) {
  auto access = analyzeGatherScatterAccess(info, indices);
  if (failed(access))
    return failure();

  auto plan = chooseTensorViewPlan(info, *access);
  if (failed(plan))
    return failure();

  switch (plan->strategy) {
  case TensorViewStrategy::ScalarTransfer:
    return emitScalarGather(b, loc, info, indices, type, space, pad);
  case TensorViewStrategy::PerIndexPayloadDMA:
    return emitPerIndexPayloadGather(b, loc, info, *access, indices, type,
                                     space, pad);
  case TensorViewStrategy::SingleAffineDMA:
    return emitSingleAffineGather();
  case TensorViewStrategy::Unknown:
  default:
    return failure();
  }
  return failure();
}

LogicalResult emitScatterStore(OpBuilder &b, Location loc,
                               const TensorViewAccessInfo &info,
                               ValueRange indices, Value source,
                               hivm::AddressSpace space) {
  auto access = analyzeGatherScatterAccess(info, indices);
  if (failed(access))
    return failure();

  auto plan = chooseTensorViewPlan(info, *access);
  if (failed(plan))
    return failure();

  switch (plan->strategy) {
  case TensorViewStrategy::ScalarTransfer:
    return emitScalarScatter(b, loc, info, indices, source);
  case TensorViewStrategy::PerIndexPayloadDMA:
    return emitPerIndexPayloadScatter(b, loc, info, *access, indices, source,
                                      space);
  case TensorViewStrategy::SingleAffineDMA:
    return emitSingleAffineScatter();
  case TensorViewStrategy::Unknown:
  default:
    return failure();
  }
  return failure();
}

} // namespace triton
} // namespace mlir
