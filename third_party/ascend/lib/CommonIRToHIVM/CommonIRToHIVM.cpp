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

//===----------------------------------------------------------------------===//
// CommonIRToHIVM — Lowers CommonIR dialect ops to HIVM (Ascend NPU) dialect
// ops.
//
// Strategy: greedy rewrite runs patterns iteratively to fixed point.
//   Step 1: tile.alloc → memref.alloc with #hivm.address_space.
//   Step 2: tile.to_tensor → UnrealizedConversionCast bridging memref to
//   tensor. Step 3: tile.copy → hivm.copy (tile.* src, on-chip DMA)
//                    or memref.copy (!tt.ptr src, GM↔local DMA; the ptr is
//                    bridged to memref via UnrealizedConversionCast)
//           tile.store_tensor → hivm.copy.
//   Step 4: tile.load/store → hivm.load/store
//   Step 5: sync ops → hivm sync ops
//===----------------------------------------------------------------------===//

#include "ascend/include/CommonIRToHIVM/Passes.h"
#include "ascend/include/CommonIRToHIVM/TensorViewAccessAnalysis.h"
#include "ascend/include/CommonIRToHIVM/TensorViewLowering.h"
#include "ascend/include/CommonIRToHIVM/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewBuilder.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_COMMONIRTVPTRLEGALIZE
#define GEN_PASS_DEF_COMMONIRTOHIVM
#include "ascend/include/CommonIRToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

// =============================================================================
// Common Helpers
// =============================================================================

static bool isOneToOne(UnrealizedConversionCastOp op) {
  return op.getInputs().size() == 1 && op->getNumResults() == 1;
}

/// Return true if \p op is an annotation.mark operation.
static bool isAnnotationMark(Operation *op) {
  return op->getName().getStringRef() == "annotation.mark";
}

static Value castDefaultMemrefToGM(Value value, Location loc,
                                   PatternRewriter &rewriter) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (!memrefTy || memrefTy.getMemorySpace())
    return value;

  auto gmSpace = hivm::AddressSpaceAttr::get(rewriter.getContext(),
                                             hivm::AddressSpace::GM);
  auto gmTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                              memrefTy.getLayout(), gmSpace);
  return rewriter.create<memref::MemorySpaceCastOp>(loc, gmTy, value);
}

static hivm::AddressSpace mapMemSpaceToHIVM(tile::MemorySpace tileSpace) {
  switch (tileSpace) {
  case tile::MemorySpace::GM:
    return hivm::AddressSpace::GM;
  case tile::MemorySpace::L1:
    return hivm::AddressSpace::L1;
  case tile::MemorySpace::L0A:
    return hivm::AddressSpace::L0A;
  case tile::MemorySpace::L0B:
    return hivm::AddressSpace::L0B;
  case tile::MemorySpace::L0C:
    return hivm::AddressSpace::L0C;
  case tile::MemorySpace::UB:
    return hivm::AddressSpace::UB;
  // Generic GPU memory spaces (Global/Shared/Local/Register) have no HIVM
  // counterpart and should never reach this lowering pass.
  case tile::MemorySpace::Global:
  case tile::MemorySpace::Shared:
  case tile::MemorySpace::Local:
  case tile::MemorySpace::Register:
    llvm_unreachable(
        "unsupported generic memory space in CommonIR-to-HIVM lowering");
  }
  llvm_unreachable("unknown CommonIR memory space");
}

static MemRefType convertBufToMemRef(tile::BufType bufTy) {
  auto shape = bufTy.getShape();
  auto elemTy = bufTy.getElementType();
  auto space = bufTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(
      llvm::SmallVector<int64_t>(shape), elemTy, MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

static MemRefType convertTensorToMemRef(tile::TensorType tensorTy) {
  auto shape = tensorTy.getShape();
  auto elemTy = tensorTy.getElementType();
  auto space = tensorTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(
      llvm::SmallVector<int64_t>(shape), elemTy, MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

/// Get the underlying memref value, converting tile/builtin tensor types to
/// memref.
static Value getAsMemRef(Value val, PatternRewriter &rewriter) {
  if (isa<MemRefType>(val.getType()))
    return val;
  // Look through UnrealizedConversionCast
  if (auto cast = val.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumResults() == 1 && cast->getNumOperands() == 1) {
      auto inner = cast->getOperand(0);
      if (isa<MemRefType>(inner.getType()))
        return inner;
    }
  }
  // Convert tile/builtin types to memref
  Type targetTy;
  if (auto bufTy = dyn_cast<tile::BufType>(val.getType()))
    targetTy = convertBufToMemRef(bufTy);
  else if (auto tensorTy = dyn_cast<tile::TensorType>(val.getType()))
    targetTy = convertTensorToMemRef(tensorTy);
  else if (auto rankedTy = dyn_cast<RankedTensorType>(val.getType())) {
    auto elemTy = rankedTy.getElementType();
    if (auto ptrElemTy = dyn_cast<triton::PointerType>(elemTy)) {
      auto gmSpace = hivm::AddressSpaceAttr::get(val.getType().getContext(),
                                                 hivm::AddressSpace::GM);
      targetTy =
          MemRefType::get(rankedTy.getShape(), ptrElemTy.getPointeeType(),
                          MemRefLayoutAttrInterface{}, gmSpace);
    } else {
      targetTy = MemRefType::get(rankedTy.getShape(), elemTy);
    }
  } else if (auto ptrTy = dyn_cast<triton::PointerType>(val.getType())) {
    // tt.ptr<tensor<...>> → memref<...>
    auto pointeeTy = ptrTy.getPointeeType();
    auto gmSpace = hivm::AddressSpaceAttr::get(val.getType().getContext(),
                                               hivm::AddressSpace::GM);
    if (auto rankedTy = dyn_cast<RankedTensorType>(pointeeTy))
      targetTy = MemRefType::get(rankedTy.getShape(), rankedTy.getElementType(),
                                 MemRefLayoutAttrInterface{}, gmSpace);
    else
      targetTy = MemRefType::get({ShapedType::kDynamic}, pointeeTy,
                                 MemRefLayoutAttrInterface{}, gmSpace);
  } else
    return val;
  return rewriter
      .create<UnrealizedConversionCastOp>(val.getLoc(), targetTy, val)
      ->getResult(0);
}

static bool isTensorOfPointer(Type ty) {
  auto rankedTy = dyn_cast<RankedTensorType>(ty);
  return rankedTy && isa<triton::PointerType>(rankedTy.getElementType());
}

// ============================================================================
// TensorView tile.load / tile.store direct lowering.
//
// A `tile.load`/`tile.store` whose operand is a `tv.tensor_view` is lowered
// here. Dense views use the dense lowering helpers; gather_scatter views use
// strategy-selected per-index payload DMA or scalar transfer.
// Both carry the explicit dst_space/src_space memory space.
// ============================================================================

/// Map TensorView padding to the access-analysis padding enum.
inline PaddingValue getPaddingValue(tv::PaddingValue value) {
  PaddingValue pad = PaddingValue::ZERO;
  switch (value) {
  case tv::PaddingValue::ZERO:
    pad = PaddingValue::ZERO;
    break;
  case tv::PaddingValue::NAN_VALUE:
    pad = PaddingValue::NAN_VALUE;
    break;
  case tv::PaddingValue::POS_INF:
    pad = PaddingValue::POS_INF;
    break;
  case tv::PaddingValue::NEG_INF:
    pad = PaddingValue::NEG_INF;
    break;
  default:
    break;
  }
  return pad;
}

// Trace a `tile.load`/`tile.store` operand back to the `tv.make_*_view` that
// produced it and gather the tile geometry. Handles both partition and strided
// encodings; for a partition view the traversal stride equals the tile size.
static TensorViewAccessInfo traceView(Value view, PatternRewriter &rewriter) {
  TensorViewAccessInfo info;
  auto viewTy = dyn_cast<tv::TensorViewType>(view.getType());
  if (!viewTy)
    return info;
  Attribute enc = viewTy.getEncoding();

  ArrayRef<int64_t> tile;
  ArrayRef<int64_t> dimMap;
  SmallVector<int64_t> dimMapStorage;
  SmallVector<int64_t> traversal;
  SmallVector<int64_t> sparseDims;
  PaddingValue padding = PaddingValue::ZERO;
  Value source;

  if (auto pv = dyn_cast_or_null<tv::PartitionViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakePartitionViewOp>();
    if (!make)
      return info;
    tile = pv.getTile();
    dimMap = pv.getDimMap();
    padding = getPaddingValue(pv.getPaddingValue());
    traversal.assign(tile.begin(), tile.end()); // step == size
    source = make.getSource();
  } else if (auto sv = dyn_cast_or_null<tv::StridedViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakeStridedViewOp>();
    if (!make)
      return info;
    tile = sv.getTile();
    dimMap = sv.getDimMap();
    padding = getPaddingValue(sv.getPaddingValue());
    ArrayRef<int64_t> ts = sv.getTraversalStrides();
    traversal.assign(ts.begin(), ts.end());
    source = make.getSource();
  } else if (auto gs = dyn_cast_or_null<tv::GatherScatterViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakeGatherScatterViewOp>();
    if (!make)
      return info;
    tile = gs.getTile();
    padding = getPaddingValue(gs.getPaddingValue());
    traversal.assign(tile.begin(), tile.end()); // contiguous step per dim
    sparseDims.assign(gs.getSparseDim().begin(), gs.getSparseDim().end());
    // A gather_scatter view keeps the base dimension order, so the tile
    // dimensions map onto the base dimensions identically.
    dimMapStorage.resize(tile.size());
    for (unsigned i = 0; i < tile.size(); ++i)
      dimMapStorage[i] = i;
    dimMap = dimMapStorage;
    source = make.getSource();
  } else {
    return info;
  }

  auto makeView = source.getDefiningOp<tv::MakeTensorViewOp>();
  auto baseTy = dyn_cast<tv::TensorViewType>(source.getType());
  if (!makeView || !baseTy)
    return info;

  if (tile.empty() || tile.size() != dimMap.size() ||
      tile.size() != traversal.size() ||
      makeView.getSizes().size() != makeView.getStrides().size() ||
      makeView.getSizes().size() != baseTy.getStrides().size())
    return info;

  Value base = getAsMemRef(makeView.getSource(), rewriter);
  if (!isa<MemRefType>(base.getType()))
    return info;
  base = castDefaultMemrefToGM(base, makeView.getLoc(), rewriter);

  for (int64_t tileSize : tile)
    if (tileSize <= 0)
      return TensorViewAccessInfo();
  for (int64_t step : traversal)
    if (step <= 0)
      return TensorViewAccessInfo();
  for (unsigned i = 0; i < dimMap.size(); ++i) {
    int64_t d = dimMap[i];
    if (d < 0 || static_cast<size_t>(d) >= makeView.getSizes().size())
      return TensorViewAccessInfo();
    for (unsigned j = 0; j < i; ++j)
      if (dimMap[j] == d)
        return TensorViewAccessInfo();
  }

  info.base = base;
  info.padding = padding;

  if (isa<tv::GatherScatterViewAttr>(enc))
    info.kind = TensorViewKind::GatherScatter;
  else if (isa<tv::StridedViewAttr>(enc))
    info.kind = TensorViewKind::Strided;
  else
    info.kind = TensorViewKind::Partition;

  info.elementType = viewTy.getElementType();
  info.tile.assign(tile.begin(), tile.end());
  info.traversal = std::move(traversal);
  info.dimMap.assign(dimMap.begin(), dimMap.end());
  info.sparseDims = std::move(sparseDims);
  info.shape.assign(makeView.getSizes().begin(), makeView.getSizes().end());
  info.staticShape.assign(baseTy.getShape().begin(), baseTy.getShape().end());
  info.strides.assign(makeView.getStrides().begin(),
                      makeView.getStrides().end());

  // The view *type*'s strides are usually all-dynamic even when the operands
  // are built from constants (a make_tensor_view on a runtime pointer carries
  // `strides=[?, ?]`), so fold each operand's constant value in here. Without
  // this, any contiguity test over `staticStrides` is unsatisfiable in
  // practice.
  info.staticStrides.assign(baseTy.getStrides().begin(),
                            baseTy.getStrides().end());
  for (unsigned d = 0; d < info.staticStrides.size(); ++d) {
    if (!ShapedType::isDynamic(info.staticStrides[d]))
      continue;
    if (auto c = getConstantIntValue(info.strides[d]))
      info.staticStrides[d] = *c;
  }

  info.ok = true;
  return info;
}

// Compute the flat element offset of the tile selected by `indices`, and
// optionally the per-dimension valid (in-bounds) lengths used for tail padding.
// The tile *starts* at `index * traversal` along each view dimension; its
// extent is `tile`. For a partition view traversal == tile, so this reduces to
// the familiar `index * tile` start. Valid length always clamps against the
// tile extent, independent of the traversal stride.
static Value emitViewGeometry(OpBuilder &builder, Location loc,
                              const TensorViewAccessInfo &info,
                              ValueRange indices,
                              SmallVectorImpl<Value> *validLengths) {
  Value offset;
  for (unsigned viewDim = 0; viewDim < info.rank(); ++viewDim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[viewDim]);
    Value index = asIndexValue(builder, loc, indices[viewDim]);
    Value step =
        builder.create<arith::ConstantIndexOp>(loc, info.traversal[viewDim]);
    Value logical = builder.create<arith::MulIOp>(loc, index, step);

    if (validLengths) {
      Value tileSize =
          builder.create<arith::ConstantIndexOp>(loc, info.tile[viewDim]);
      Value extent = asIndexValue(builder, loc, info.shape[baseDim]);
      Value remaining = builder.create<arith::SubIOp>(loc, extent, logical);
      Value length = builder.create<arith::MinSIOp>(loc, remaining, tileSize);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      validLengths->push_back(
          builder.create<arith::MaxSIOp>(loc, length, zero));
    }

    Value stride = asIndexValue(builder, loc, info.strides[baseDim]);
    Value physical = builder.create<arith::MulIOp>(loc, logical, stride);
    offset = offset ? builder.create<arith::AddIOp>(loc, offset, physical)
                    : physical;
  }
  return offset;
}

// Build the GM-side memref for one tile via reinterpret_cast. The tile shape is
// `tile` (the traversal stride only moved the start offset, not the extent),
// so this helper is identical for partition and strided views.
static Value emitViewGmTile(OpBuilder &builder, Location loc,
                            const TensorViewAccessInfo &info, Value offset) {
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> staticStrides;
  for (unsigned viewDim = 0; viewDim < info.rank(); ++viewDim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[viewDim]);
    sizes.push_back(builder.getIndexAttr(info.tile[viewDim]));
    int64_t stride = info.staticStrides[baseDim];
    staticStrides.push_back(stride);
    if (stride == ShapedType::kDynamic)
      strides.push_back(info.strides[baseDim]);
    else
      strides.push_back(builder.getIndexAttr(stride));
  }

  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, staticStrides);
  auto gmSpace =
      hivm::AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::GM);
  auto gmTy = MemRefType::get(info.tile, info.elementType, layout, gmSpace);
  return builder.create<memref::ReinterpretCastOp>(
      loc, gmTy, info.base, OpFoldResult(offset), sizes, strides);
}

// Restrict a full tile memref to its in-bounds prefix along every dimension so
// only valid lanes are moved; the padding fill covers the rest.
static Value emitPrefixSubview(OpBuilder &builder, Location loc, Value source,
                               ValueRange validLengths) {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  for (Value length : validLengths) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(OpFoldResult(length));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
}

// Resolve the on-chip memory space carried by a tile.load/tile.store into a
// HIVM address space. A null/absent attribute means the DSL did not specify a
// space, so we default to UB (the Vector working buffer). Generic GPU spaces
// are rejected: they never occur on the Ascend lowering path.
static hivm::AddressSpace
resolveTileSpace(std::optional<tile::MemorySpace> space) {
  if (!space)
    return hivm::AddressSpace::UB;
  return mapMemSpaceToHIVM(*space);
}

// Build the constant used to fill a tile's out-of-bounds lanes according to the
// view's padding value: the tail lanes of a dense boundary tile or the
// out-of-range lanes of a gather_scatter gather. Float element types support
// the IEEE special pads (nan / +inf / -inf); integers only support zero.
static Value emitPadConstant(OpBuilder &builder, Location loc, Type elementType,
                             PaddingValue padding) {
  if (auto floatTy = dyn_cast<FloatType>(elementType)) {
    const llvm::fltSemantics &sem = floatTy.getFloatSemantics();
    llvm::APFloat value = llvm::APFloat::getZero(sem);
    switch (padding) {
    case PaddingValue::NAN_VALUE:
      value = llvm::APFloat::getNaN(sem);
      break;
    case PaddingValue::POS_INF:
      value = llvm::APFloat::getInf(sem, /*Negative=*/false);
      break;
    case PaddingValue::NEG_INF:
      value = llvm::APFloat::getInf(sem, /*Negative=*/true);
      break;
    default:
      break;
    }
    return builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(elementType, value));
  }
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getZeroAttr(elementType));
}

// A tile.load/tile.store lowers through the TensorView path when its operand
// carries a partition, strided, or gather_scatter encoding and the index
// count matches the tile rank. The dense encodings share the contiguous DMA
// lowering; a gather_scatter view adds the discrete access path.
static bool isViewLowerable(tv::TensorViewType viewTy, size_t numIndices) {
  Attribute enc = viewTy.getEncoding();
  if (auto pv = dyn_cast_or_null<tv::PartitionViewAttr>(enc))
    return numIndices == pv.getTile().size();
  if (auto sv = dyn_cast_or_null<tv::StridedViewAttr>(enc))
    return numIndices == sv.getTile().size();
  if (auto gs = dyn_cast_or_null<tv::GatherScatterViewAttr>(enc))
    return numIndices == gs.getTile().size();
  return false;
}

// A dense tile is fully in bounds when every dimension's valid length
// (computed by emitViewGeometry) covers the whole tile extent. The base
// view's shape is a runtime value, so this can only be decided at runtime,
// not from the static tile size alone.
static Value emitAllInBounds(OpBuilder &b, Location loc,
                             const TensorViewAccessInfo &info,
                             ValueRange validLengths) {
  Value allIn = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
  for (unsigned d = 0; d < info.rank(); ++d) {
    Value tileSize = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    Value full = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         validLengths[d], tileSize);
    allIn = b.create<arith::AndIOp>(loc, allIn, full);
  }
  return allIn;
}

// Load a dense (partition/strided) tile, choosing at runtime between a
// whole-tile DMA and emitElementwiseFill's padded gather. Whether the tile
// stays in bounds is a runtime property of the base view's shape, so scf.if
// picks the branch once per tile invocation rather than at compile time. Both
// branches fill the same UB buffer, which keeps the scf.if result-free and
// gives the tile a single definition regardless of which one ran. A boundary
// tile takes the gather branch even though the view is dense, because the
// in-bounds region of a multi-dim tile is generally L-shaped and cannot be
// expressed as a single rectangular subview of the whole-tile DMA.
static Value emitDenseTileLoad(OpBuilder &b, Location loc,
                               const TensorViewAccessInfo &info,
                               ValueRange indices, Value offset,
                               ValueRange validLengths,
                               hivm::AddressSpace dstSpace,
                               RankedTensorType resultTy, Value pad) {
  Value ub = b.create<memref::AllocOp>(
      loc,
      makeSpaceMemref(b.getContext(), info.tile, info.elementType, dstSpace));
  Value allInBounds = emitAllInBounds(b, loc, info, validLengths);
  b.create<scf::IfOp>(
      loc, allInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value gm = emitViewGmTile(nested, nestedLoc, info, offset);
        nested.create<hivm::LoadOp>(nestedLoc, TypeRange{}, gm, ub);
        nested.create<scf::YieldOp>(nestedLoc);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        emitElementwiseFill(nested, nestedLoc, info, indices, ub, pad);
        nested.create<scf::YieldOp>(nestedLoc);
      });
  return b.create<bufferization::ToTensorOp>(loc, resultTy, ub, true, false);
}

// A gather_scatter view pairs each sparse dimension with a tensor-valued index
// and every other dimension with a scalar index; a dense view (empty
// sparseDims) therefore expects all-scalar indices. Reject a mismatched operand
// so the element-access emitters can assume the pairing holds.
static bool viewIndicesWellTyped(const TensorViewAccessInfo &info,
                                 ValueRange indices) {
  for (unsigned d = 0; d < indices.size(); ++d) {
    bool isSparse =
        llvm::is_contained(info.sparseDims, static_cast<int64_t>(d));
    bool isTensor = isa<RankedTensorType>(indices[d].getType());
    if (isSparse != isTensor)
      return false;
  }
  return true;
}

// lowerViewLoad/lowerViewStore below call emitGatherLoad/emitScatterStore from
// TensorViewLowering.cpp; nothing reaches the emitters in the .inc any more
// (verified: zero references to emitPerIndexPayloadGather,
// emitSingleAffineGatherDMA or emitDensePayloadInBounds from this file).
// Including it only compiled a second, divergent copy of helpers that share
// names with the live ones -- edits to one silently did not affect the other.
//
// The file is kept rather than deleted because it holds the only working
// implementation of the affine fast path: emitAffineGmTile (which folds the
// index step into the sparse dim's stride) and emitAffineRunInBounds (which
// range-checks both ends of a run once per tile). Those are what
// emitSingleAffineGather/Scatter in TensorViewLowering.cpp need when the stubs
// there are filled in. See documents/commonir/tensorview_known_issues.md,
// Issues 1 and 3.

// Lower a `tile.load` off a TensorView. Dense views use emitDenseTileLoad;
// gather_scatter views use emitGatherLoad to select payload DMA or scalar
// transfer with bounds handling. The tile geometry comes from `traceView`.

static LogicalResult lowerViewLoad(tile::LoadOp op, PatternRewriter &rewriter) {
  auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultTy)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

  TensorViewAccessInfo info = traceView(op.getSrc(), rewriter);
  if (!info.ok || op.getIndices().size() != info.rank() ||
      resultTy.getRank() != static_cast<int64_t>(info.rank()) ||
      resultTy.getElementType() != info.elementType ||
      resultTy.getShape() != ArrayRef<int64_t>(info.tile))
    return rewriter.notifyMatchFailure(op, "unsupported TensorView load shape");
  if (!viewIndicesWellTyped(info, op.getIndices()))
    return rewriter.notifyMatchFailure(
        op,
        "TensorView load: sparse dims need tensor indices, dense dims scalar");

  Location loc = op.getLoc();
  // Only the GM -> UB DMA is lowered here; other on-chip spaces
  // (L1/L0A/L0B/L0C) are unsupported.
  hivm::AddressSpace space = resolveTileSpace(op.getDstSpace());
  if (space != hivm::AddressSpace::UB)
    return rewriter.notifyMatchFailure(
        op, "TensorView load: dst_space other than UB is unsupported");

  Value pad = emitPadConstant(rewriter, loc, info.elementType, info.padding);

  // PartitionView/StridedView
  if (!info.isSparse()) {
    SmallVector<Value> validLengths;
    Value offset =
        emitViewGeometry(rewriter, loc, info, op.getIndices(), &validLengths);
    Value tensor =
        emitDenseTileLoad(rewriter, loc, info, op.getIndices(), offset,
                          validLengths, space, resultTy, pad);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  // GatherScatterView selects per-index payload DMA when the strategy permits,
  // otherwise scalar transfer. Out-of-bounds elements take the padding value.
  // emitGatherLoad covers every lowerable gather: its own ScalarTransfer
  // branch is the element-wise path, so there is no fallback to add here.
  // A failure means the view is not lowerable at all, which leaves the
  // tile.load for whoever can handle it rather than being quietly dropped.
  // GatherScatterView
  auto gathered = emitGatherLoad(rewriter, loc, info, op.getIndices(), resultTy,
                                 space, pad);
  if (failed(gathered))
    return rewriter.notifyMatchFailure(op, "TensorView load: view is not "
                                           "lowerable as a gather");
  rewriter.replaceOp(op, *gathered);
  return success();
}

// Lower a `tile.store` into a TensorView. GatherScatterView selects per-index
// payload DMA or scalar transfer and leaves out-of-bounds elements unwritten;
// dense views use the dense store path below.
static LogicalResult lowerViewStore(tile::StoreOp op,
                                    PatternRewriter &rewriter) {
  auto valueTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!valueTy)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor source");

  TensorViewAccessInfo info = traceView(op.getDst(), rewriter);
  if (!info.ok || op.getIndices().size() != info.rank() ||
      valueTy.getRank() != static_cast<int64_t>(info.rank()) ||
      valueTy.getElementType() != info.elementType ||
      valueTy.getShape() != ArrayRef<int64_t>(info.tile))
    return rewriter.notifyMatchFailure(op,
                                       "unsupported TensorView store shape");
  if (!viewIndicesWellTyped(info, op.getIndices()))
    return rewriter.notifyMatchFailure(
        op,
        "TensorView store: sparse dims need tensor indices, dense dims scalar");

  Location loc = op.getLoc();
  // Only the UB -> GM DMA is lowered here; other on-chip spaces
  // (L1/L0A/L0B/L0C) are unsupported.
  hivm::AddressSpace srcSpace = resolveTileSpace(op.getSrcSpace());
  if (srcSpace != hivm::AddressSpace::UB)
    return rewriter.notifyMatchFailure(
        op, "TensorView store: src_space other than UB is unsupported");

  // A gather_scatter view writes valid payloads or scalar elements to their
  // indexed coordinates. As on the load side, emitScatterStore's
  // ScalarTransfer branch *is* the element-wise scatter, so a failure means
  // the view is not lowerable rather than "take the slow path".
  if (!info.sparseDims.empty()) {
    if (failed(emitScatterStore(rewriter, loc, info, op.getIndices(),
                                op.getSrc(), srcSpace)))
      return rewriter.notifyMatchFailure(op, "TensorView store: view is not "
                                             "lowerable as a scatter");
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<Value> validLengths;
  Value offset =
      emitViewGeometry(rewriter, loc, info, op.getIndices(), &validLengths);
  Value gm = emitViewGmTile(rewriter, loc, info, offset);
  Value ub = rewriter.create<memref::AllocOp>(
      loc, makeSpaceMemref(rewriter.getContext(), info.tile, info.elementType,
                           srcSpace));
  // Stage the whole tile in UB with static shapes and clip only the UB -> GM
  // DMA to the in-bounds prefix. Slicing the source tensor to the prefix first
  // would hand the rest of the pipeline a dynamically shaped tensor, which
  // fails to compile for any computed store value.
  auto materialize = rewriter.create<bufferization::MaterializeInDestinationOp>(
      loc, op.getSrc(), ub);
  materialize->setAttr("writable", rewriter.getUnitAttr());
  Value gmSub = emitPrefixSubview(rewriter, loc, gm, validLengths);
  Value ubSub = emitPrefixSubview(rewriter, loc, ub, validLengths);
  rewriter.create<hivm::StoreOp>(loc, TypeRange{}, ubSub, gmSub);
  rewriter.eraseOp(op);
  return success();
}

static void lowerScatteredLoad(tile::CopyOp op, Value ptr, Value buf,
                               PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value loaded = rewriter.create<triton::LoadOp>(
      loc, ptr, Value(), Value(), ArrayRef<int32_t>{},
      std::optional<triton::PaddingOption>(), triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);

  auto ty = cast<RankedTensorType>(loaded.getType());
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value mem = rewriter.create<bufferization::ToMemrefOp>(
      loc, MemRefType::get(ty.getShape(), ty.getElementType()), loaded);
#else
  Value mem = rewriter.create<bufferization::ToBufferOp>(
      loc, MemRefType::get(ty.getShape(), ty.getElementType()), loaded);
#endif

  rewriter.create<hivm::CopyOp>(loc, TypeRange{}, mem,
                                getAsMemRef(buf, rewriter));
  rewriter.eraseOp(op);
}

static void lowerScatteredStore(tile::CopyOp op, Value buf, Value ptr,
                                PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value mem = getAsMemRef(buf, rewriter);
  auto m = cast<MemRefType>(mem.getType());

  Value t = rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(m.getShape(), m.getElementType()), mem, true,
      false);

  rewriter.replaceOpWithNewOp<triton::StoreOp>(
      op, ptr, t, Value(), ArrayRef<int32_t>{}, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);
}

static bool isTritonPointerLike(Type ty) {
  if (isa<triton::PointerType>(ty))
    return true;
  if (auto rankedTy = dyn_cast<RankedTensorType>(ty))
    return isa<triton::PointerType>(rankedTy.getElementType());
  return false;
}

static hivm::PIPE mapPipe(int64_t tilePipe) {
  switch (static_cast<tile::Pipe>(tilePipe)) {
  case tile::Pipe::PIPE_M:
    return hivm::PIPE::PIPE_M;
  case tile::Pipe::PIPE_V:
    return hivm::PIPE::PIPE_V;
  case tile::Pipe::PIPE_MTE1:
    return hivm::PIPE::PIPE_MTE1;
  case tile::Pipe::PIPE_MTE2:
    return hivm::PIPE::PIPE_MTE2;
  case tile::Pipe::PIPE_MTE3:
    return hivm::PIPE::PIPE_MTE3;
  case tile::Pipe::PIPE_FIX:
    return hivm::PIPE::PIPE_FIX;
  case tile::Pipe::PIPE_S:
    return hivm::PIPE::PIPE_S;
  }
  llvm_unreachable("unknown CommonIR pipe");
}

static hivm::EVENT mapEvent(int64_t tileEvent) {
  return static_cast<hivm::EVENT>(tileEvent);
}

// =============================================================================
// Step 1: tile.alloc → memref.alloc + #hivm.address_space
// =============================================================================
struct TileAllocToMemRef : OpRewritePattern<tile::AllocOp> {
  using OpRewritePattern<tile::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::AllocOp op,
                                PatternRewriter &rewriter) const final {
    auto bufTy = op.getResult().getType();
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, convertBufToMemRef(bufTy));
    return success();
  }
};

// =============================================================================
// step 1.5
// =============================================================================

struct TileSubviewToMemrefSubview : public OpRewritePattern<tile::SubViewOp> {
  using OpRewritePattern<tile::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Convert the source buffer and deduce the result type
    Value sourceMemRef = getAsMemRef(op.getOperand(0), rewriter);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    auto sourceMemRefType = sourceMemRef.getType().cast<MemRefType>();
#else
    auto sourceMemRefType = mlir::cast<MemRefType>(sourceMemRef.getType());
#endif

    // 2. Map dynamic offsets directly to OpFoldResult
    SmallVector<OpFoldResult> mixedOffsets = llvm::to_vector<4>(llvm::map_range(
        op.getOffsets(), [](Value v) { return OpFoldResult(v); }));

    // 3. Unpack static sizes and strides into OpFoldResult arrays
    SmallVector<OpFoldResult> mixedSizes = getAsOpFoldResult(op.getSizesAttr());
    SmallVector<OpFoldResult> mixedStrides =
        getAsOpFoldResult(op.getStridesAttr());

    auto targetShape = op.getType().getShape();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    auto inferredType = memref::SubViewOp::inferRankReducedResultType(
                            targetShape, sourceMemRefType, mixedOffsets,
                            mixedSizes, mixedStrides)
                            .cast<MemRefType>();
#else
    auto inferredType =
        mlir::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            targetShape, sourceMemRefType, mixedOffsets, mixedSizes,
            mixedStrides));
#endif

    // 4. Replace the old op with the standard memref.subview
    auto memrefSubview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), inferredType, sourceMemRef, mixedOffsets, mixedSizes,
        mixedStrides);

    rewriter.replaceOp(op, memrefSubview.getResult());

    return success();
  }
};

// =============================================================================
// Step 2: tile.to_tensor → UnrealizedConversionCast (memref → tensor)
// =============================================================================
struct TileToTensorEliminate : OpRewritePattern<tile::ToTensorOp> {
  using OpRewritePattern<tile::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::ToTensorOp op,
                                PatternRewriter &rewriter) const final {
    // The src is now memref (from Step 1).  The result type is tensor<>.
    // Bridge the gap with UnrealizedConversionCast: memref → tensor.
    Value src = op.getOperand();
    auto resultTy = op.getResult().getType();
    if (op->getAttr("writable")) {
      // The caller (tle.dsa.to_tensor defaults to writable=True) expects an
      // in-place view: custom ops use such a tensor as a DPS out and write
      // the results back into the source buffer. Lower it to a writable
      // bufferization.to_tensor; a plain cast would make One-Shot
      // Bufferization copy the out into a fresh UB allocation that
      // hivm-plan-memory cannot plan, which ends up as a `call @malloc` in
      // the device code (AICore link failure) and a broken dataflow.
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          op, cast<RankedTensorType>(resultTy), src,
          /*restrict=*/true, /*writable=*/true);
      return success();
    }
    if (src.getType() != resultTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(),
                                                              resultTy, src);
      rewriter.replaceOp(op, cast->getResult(0));
    } else {
      rewriter.replaceOp(op, src);
    }
    return success();
  }
};

// =============================================================================
// Step 3: tile.copy → hivm.copy / memref.copy
//   - tile.* source  → hivm.copy (on-chip buffer DMA)
//   - !tt.ptr source → memref.copy (GM→local DMA; later passes lower it)
// =============================================================================
struct TileCopyToHIVM : OpRewritePattern<tile::CopyOp> {
  using OpRewritePattern<tile::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);

    if (isTensorOfPointer(src.getType())) {
      lowerScatteredLoad(op, src, dst, rewriter);
      return success();
    }
    if (isTensorOfPointer(dst.getType())) {
      lowerScatteredStore(op, src, dst, rewriter);
      return success();
    }

    Value srcMem = getAsMemRef(src, rewriter);
    Value dstMem = getAsMemRef(dst, rewriter);
    if (isTritonPointerLike(src.getType())) {
      rewriter.replaceOpWithNewOp<memref::CopyOp>(op, srcMem, dstMem);
      return success();
    }
    rewriter.create<hivm::CopyOp>(op.getLoc(), TypeRange{}, srcMem, dstMem);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TileStoreTensorToHIVM : OpRewritePattern<tile::StoreTensorOp> {
  using OpRewritePattern<tile::StoreTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::StoreTensorOp op,
                                PatternRewriter &rewriter) const final {
    Value dstMem = getAsMemRef(op.getOperand(1), rewriter);
    Value srcMem = op.getOperand(0);
    if (!isa<MemRefType>(srcMem.getType())) {
      auto tensorTy = dyn_cast<RankedTensorType>(srcMem.getType());
      if (!tensorTy)
        return failure();
      auto memrefTy =
          MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
      srcMem = rewriter.create<bufferization::ToMemrefOp>(op.getLoc(), memrefTy,
                                                          srcMem);
#else
      srcMem = rewriter.create<bufferization::ToBufferOp>(op.getLoc(), memrefTy,
                                                          srcMem);
#endif
    }

    rewriter.create<hivm::CopyOp>(op.getLoc(), TypeRange{}, srcMem, dstMem);
    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// Step 4: tile.load → hivm.load
// =============================================================================
struct TileLoadToHIVM : OpRewritePattern<tile::LoadOp> {
  using OpRewritePattern<tile::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::LoadOp op,
                                PatternRewriter &rewriter) const final {
    // TensorView operand -> direct lowering (dense DMA or gather_scatter
    // gather).
    if (auto viewTy = dyn_cast<tv::TensorViewType>(op.getSrc().getType())) {
      if (isViewLowerable(viewTy, op.getIndices().size()))
        return lowerViewLoad(op, rewriter);
      return failure();
    }
    auto resultTy = op.getResult().getType();
    if (auto t = dyn_cast<tile::TensorType>(resultTy)) {
      auto memrefTy = convertTensorToMemRef(t);
      rewriter.replaceOpWithNewOp<hivm::LoadOp>(op, memrefTy, op.getSrc());
      return success();
    }
    return failure();
  }
};

// =============================================================================
// Step 4: tile.store → hivm.store
// =============================================================================
struct TileStoreToHIVM : OpRewritePattern<tile::StoreOp> {
  using OpRewritePattern<tile::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::StoreOp op,
                                PatternRewriter &rewriter) const final {
    // TensorView destination -> direct lowering (dense DMA or gather_scatter
    // scatter).
    if (auto viewTy = dyn_cast<tv::TensorViewType>(op.getDst().getType())) {
      if (isViewLowerable(viewTy, op.getIndices().size()))
        return lowerViewStore(op, rewriter);
      return failure();
    }
    if (!op.getIndices().empty())
      return failure();
    Value src = getAsMemRef(op.getSrc(), rewriter);
    rewriter.replaceOpWithNewOp<hivm::StoreOp>(op, Type(), src, op.getDst());
    return success();
  }
};

// =============================================================================
// Step 5: Sync ops
// =============================================================================
struct TileSetFlagToHIVM : OpRewritePattern<tile::SetFlagOp> {
  using OpRewritePattern<tile::SetFlagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::SetFlagOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SetFlagOp>(
        op,
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx,
                             mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TileWaitFlagToHIVM : OpRewritePattern<tile::WaitFlagOp> {
  using OpRewritePattern<tile::WaitFlagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::WaitFlagOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::WaitFlagOp>(
        op,
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx,
                             mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TilePipeBarrierToHIVM : OpRewritePattern<tile::PipeBarrierOp> {
  using OpRewritePattern<tile::PipeBarrierOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::PipeBarrierOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::PipeBarrierOp>(
        op,
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getPipe()))));
    return success();
  }
};

struct TileCubeWaitToHIVM : OpRewritePattern<tile::CubeWaitOp> {
  using OpRewritePattern<tile::CubeWaitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::CubeWaitOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SyncBlockWaitOp>(
        op, hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
        OpFoldResult(rewriter.getIndexAttr(0)));
    return success();
  }
};

struct TileGmOffsetToHIVM : OpRewritePattern<tile::GmOffsetOp> {
  using OpRewritePattern<tile::GmOffsetOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::GmOffsetOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value result = op.getBase();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    for (size_t i = 0; i < indices.size(); ++i) {
      Value off = rewriter.create<arith::MulIOp>(loc, indices[i], strides[i]);
      result = rewriter.create<arith::AddIOp>(loc, result, off);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern A: !tt.ptr<tensor<>> -> memref<>
//===----------------------------------------------------------------------===//
struct LowerPtrTensorCastToMemref
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto ptrTy = dyn_cast<triton::PointerType>(in.getType());
    if (!ptrTy)
      return failure();
    auto tensorPointee = dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
    if (!tensorPointee)
      return failure();
    auto outMemref = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (!outMemref)
      return failure();
    if (outMemref.getShape() != tensorPointee.getShape() ||
        outMemref.getElementType() != tensorPointee.getElementType())
      return failure();

    Location loc = op.getLoc();
    // %t = tt.load %ptr : !tt.ptr<tensor<...>> (tensor-pointer overload)
    Value loaded = rewriter.create<triton::LoadOp>(
        loc, in, /*boundaryCheck=*/ArrayRef<int32_t>{},
        /*padding=*/std::optional<triton::PaddingOption>(),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
    // %m = bufferization.to_memref %t
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    Value asMemref =
        rewriter.create<bufferization::ToMemrefOp>(loc, outMemref, loaded);
#else
    Value asMemref =
        rewriter.create<bufferization::ToBufferOp>(loc, outMemref, loaded);
#endif
    rewriter.replaceOp(op, asMemref);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern B: memref<..., #space> -> tensor<...>
//===----------------------------------------------------------------------===//
struct LowerMemrefCastToTensor
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto memrefTy = dyn_cast<MemRefType>(in.getType());
    if (!memrefTy)
      return failure();
    auto tensorOut = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorOut)
      return failure();
    if (memrefTy.getShape() != tensorOut.getShape() ||
        memrefTy.getElementType() != tensorOut.getElementType())
      return failure();

    Location loc = op.getLoc();
    // %t = bufferization.to_tensor %m restrict
    Value asTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorOut, in, /*restrict=*/true, /*writable=*/false);
    rewriter.replaceOp(op, asTensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern C (utility): simplify chains of one-to-one unrealized casts whose
// outer source type equals the outer result type.  This catches cases like
//   %a = unrealized_conversion_cast %x : memref -> tensor
//   %b = unrealized_conversion_cast %a : tensor -> memref
// which arise after the linalg pass partially converts surrounding ops.
//===----------------------------------------------------------------------===//
struct FoldRoundtripCast : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto prev = in.getDefiningOp<UnrealizedConversionCastOp>();
    if (!prev || !isOneToOne(prev))
      return failure();
    Value origin = prev.getInputs().front();
    if (origin.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, origin);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Rewrite pattern: fold staging alloc + copy1 + to_tensor + copy2
//===----------------------------------------------------------------------===//
struct FoldStagingCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value stage = copyOp.getSource();
    auto stageCast = stage.getDefiningOp<memref::MemorySpaceCastOp>();
    if (stageCast)
      stage = stageCast.getSource();
    Value dst = copyOp.getTarget();

    // ① Destination must have an explicit memory space (on-chip buffer).
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType || !dstType.getMemorySpace())
      return failure();

    // ② Source must be a staging memref.alloc in default/GM address space.
    auto stageAlloc = stage.getDefiningOp<memref::AllocOp>();
    if (!stageAlloc)
      return failure();
    auto stageType = stageAlloc.getType();
    if (auto stageSpace = stageType.getMemorySpace()) {
      auto stageAddr = dyn_cast<hivm::AddressSpaceAttr>(stageSpace);
      if (!stageAddr || stageAddr.getAddressSpace() != hivm::AddressSpace::GM)
        return failure();
    }

    // ③ Enumerate the source copy: memref.copy %src, %stage.
    //    Collect all users of %stage to verify the expected pattern.
    memref::CopyOp srcCopy;
    Operation *stageMark = nullptr;
    bufferization::ToTensorOp toTensor;
    Operation *tensorMark = nullptr;
    SmallVector<Operation *> otherUses;

    for (Operation *user : stage.getUsers()) {
      if (auto c = dyn_cast<memref::CopyOp>(user)) {
        if (c.getTarget() == stage) {
          if (srcCopy)
            return failure(); // multiple incoming copies — ambiguous
          srcCopy = c;
        } else if (c == copyOp.getOperation()) {
          continue; // the outgoing copy we are folding
        } else {
          otherUses.push_back(user);
        }
      } else if (isAnnotationMark(user)) {
        if (stageMark)
          otherUses.push_back(user);
        else
          stageMark = user;
      } else if (stageCast && user == stageCast.getOperation()) {
        continue;
      } else if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (toTensor)
          otherUses.push_back(user);
        else
          toTensor = tt;
      } else {
        otherUses.push_back(user);
      }
    }

    // srcCopy is mandatory; toTensor is optional (Flavour A has it, B doesn't).
    if (!srcCopy || !otherUses.empty())
      return failure();

    // Collect annotation.mark users of the tensor value (only if toTensor
    // exists — Flavour A path).
    if (toTensor) {
      Value tensorVal = toTensor.getResult();
      for (Operation *user : tensorVal.getUsers()) {
        if (isAnnotationMark(user) && !tensorMark)
          tensorMark = user;
      }
    }

    Location loc = copyOp.getLoc();
    Value src = castDefaultMemrefToGM(srcCopy.getSource(), loc, rewriter);
    Value dstCbuf = dst;

    // ④ Create the merged copy: memref.copy %src, %dst(#space)
    rewriter.create<memref::CopyOp>(loc, src, dstCbuf);

    // ⑤ If there was an annotation.mark on %stage, move it to %dst.
    if (stageMark) {
      OperationState state(loc, "annotation.mark");
      state.addOperands(dstCbuf);
      for (auto &namedAttr : stageMark->getAttrs())
        state.addAttribute(namedAttr.getName(), namedAttr.getValue());
      rewriter.create(state);
    }

    // ⑥ For Flavour A: drop the address space via memory_space_cast and
    //    re-create the to_tensor pointing at the cbuf alloc.
    //    For Flavour B: nothing to do — the tensor path never existed.
    if (toTensor) {
      auto genericType =
          MemRefType::get(dstType.getShape(), dstType.getElementType());
      Value cast =
          rewriter.create<memref::MemorySpaceCastOp>(loc, genericType, dstCbuf);
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          toTensor.getLoc(), toTensor.getType(), cast, toTensor.getRestrict(),
          toTensor.getWritable());

      // ⑦ If there was an annotation.mark on the tensor value, re-create it.
      if (tensorMark) {
        OperationState state(loc, "annotation.mark");
        state.addOperands(tensor);
        for (auto &namedAttr : tensorMark->getAttrs())
          state.addAttribute(namedAttr.getName(), namedAttr.getValue());
        rewriter.create(state);
      }

      rewriter.replaceOp(toTensor, tensor);
    }

    // ⑧ Erase the dead ops.
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(srcCopy);
    if (stageCast)
      rewriter.eraseOp(stageCast);
    if (stageMark)
      rewriter.eraseOp(stageMark);
    if (tensorMark)
      rewriter.eraseOp(tensorMark);

    return success();
  }
};

// =============================================================================
// view base pointer signature legalization
// =============================================================================

// Convert a view base pointer to memref<?xT, GM> (a ranked pointee keeps its
// shape; otherwise a dynamic 1-D memref). The GM address space is preserved so
// the downstream lowering sees a global-memory buffer.
static MemRefType convertPtrToMemRef(Type ptrTy) {
  auto pointeeTy = tv::getPtrPointeeType(ptrTy);
  auto *ctx = pointeeTy.getContext();
  auto gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
  if (auto rankedTy = dyn_cast<RankedTensorType>(pointeeTy))
    return MemRefType::get(rankedTy.getShape(), rankedTy.getElementType(),
                           MemRefLayoutAttrInterface{}, gmSpace);
  return MemRefType::get({ShapedType::kDynamic}, pointeeTy,
                         MemRefLayoutAttrInterface{}, gmSpace);
}

// Whether \p arg is a pointer this pass should convert: every use has to be a
// view construction. Deciding here rather than in the frontend builder is what
// makes the check sound: the whole body is already built, so the uses are all
// visible.
//
// A pointer that also feeds ordinary pointer arithmetic is left alone, and the
// mixed case consequently does not compile. Converting it was measured and is
// not the fix: the entry argument becomes a GM memref bridged back with an
// unrealized cast, which reads correctly after this pass, but the cast does not
// survive for the arithmetic users -- `tt.splat` ends up back on a bare
// `tt.ptr` and TritonToUnstructure fails. Not converting instead leaves
// `traceView` to bridge `tt.ptr` to memref at the point of use, which
// TritonToLinalg reports as a live materialization. Both directions fail, so
// supporting mixed use means giving the view its own buffer without retyping
// the argument.
static bool isExclusiveViewBasePtr(BlockArgument arg) {
  if (!isa<triton::PointerType>(arg.getType()))
    return false;
  if (arg.use_empty())
    return false;
  return llvm::all_of(arg.getUsers(), [](Operation *user) {
    return isa<tv::MakeTensorViewOp>(user);
  });
}

// Rewrite each kernel argument of \p func that only feeds view constructions to
// memref<?xT, GM>. The retyped entry-block argument is bridged back to its
// original pointer type with an unrealized_conversion_cast so that the
// make_tensor_view ops still consuming it keep verifying; the CommonIR-to-HIVM
// lowering recovers the memref through that cast.
//
// Without this the cast runs the other way: `getAsMemRef` would bridge a
// `tt.ptr` argument to a memref at the point of use, and nothing downstream
// resolves that direction -- TritonToLinalg reports the materialization as
// still live. Converting the argument itself is what gives the view a GM
// buffer.
static LogicalResult legalizeViewBasePtrSignature(triton::FuncOp func,
                                                  ModuleOp module) {
  if (func.empty())
    return success();

  Block &entry = func.front();
  SmallVector<Type> newInputs(func.getFunctionType().getInputs());
  SmallVector<BlockArgument> toConvert;
  for (BlockArgument arg : entry.getArguments()) {
    if (!isExclusiveViewBasePtr(arg))
      continue;
    toConvert.push_back(arg);
    newInputs[arg.getArgNumber()] = convertPtrToMemRef(arg.getType());
  }

  if (toConvert.empty())
    return success();

  // Changing the signature would break any tt.call that targets this function:
  // the call operands keep their pointer type and no longer match. Converting
  // them here is not a fix either -- a cast from the caller's pointer to a
  // memref is the direction nothing downstream resolves. Entry kernels have no
  // callers, and an inlinable callee is already gone by this point, so the
  // remaining case is a noinline helper that builds a view on its own pointer
  // parameter. Report that instead of emitting a module that fails to verify.
  if (!SymbolTable::symbolKnownUseEmpty(func, module))
    return func.emitError()
           << "cannot legalize view base pointer arguments of '"
           << func.getName()
           << "': the function is called, and its call sites would keep the "
              "original pointer type. Allow this helper to be inlined (drop "
              "`noinline`) or build the view in the caller";

  func.setFunctionType(FunctionType::get(func.getContext(), newInputs,
                                         func.getFunctionType().getResults()));

  OpBuilder builder(&entry, entry.begin());
  for (BlockArgument arg : toConvert) {
    Type origTy = arg.getType();
    arg.setType(newInputs[arg.getArgNumber()]);
    auto cast =
        builder.create<UnrealizedConversionCastOp>(func.getLoc(), origTy, arg);
    arg.replaceAllUsesExcept(cast.getResult(0), cast);
  }
  return success();
}

namespace {
struct CommonIRTvPtrLegalizePass
    : public mlir::triton::impl::CommonIRTvPtrLegalizeBase<
          CommonIRTvPtrLegalizePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto result = module.walk([&](triton::FuncOp func) {
      return failed(legalizeViewBasePtrSignature(func, module))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCommonIRTvPtrLegalizePass() {
  return std::make_unique<CommonIRTvPtrLegalizePass>();
}

// =============================================================================
// Pass
// =============================================================================
namespace {
struct CommonIRToHIVMPass
    : public mlir::triton::impl::CommonIRToHIVMBase<CommonIRToHIVMPass> {
  void runOnOperation() override;
};
} // namespace

void CommonIRToHIVMPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy rewrite to iteratively apply patterns to fixed point.
  // This avoids the type-conversion complexity of the dialect conversion
  // framework: tile.alloc → memref.alloc replaces !tile.buf with memref,
  // then tile.copy (now seeing memref operands) → hivm.copy, etc.

// apply pattern separately to ensure their relative order
// template lambda is a C++ 20 feature, so we'll have to use MACRO
#define APPLY_REWRITE_PATTERN(...)                                             \
  {                                                                            \
    RewritePatternSet patterns(&getContext());                                 \
    patterns.add<__VA_ARGS__>(&getContext());                                  \
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {   \
      return signalPassFailure();                                              \
    }                                                                          \
  }

  APPLY_REWRITE_PATTERN(TileAllocToMemRef);
  APPLY_REWRITE_PATTERN(TileSubviewToMemrefSubview);
  APPLY_REWRITE_PATTERN(TileToTensorEliminate);
  APPLY_REWRITE_PATTERN(TileCopyToHIVM);
  APPLY_REWRITE_PATTERN(TileStoreTensorToHIVM);
  APPLY_REWRITE_PATTERN(TileLoadToHIVM);
  APPLY_REWRITE_PATTERN(TileStoreToHIVM);
  APPLY_REWRITE_PATTERN(TileSetFlagToHIVM);
  APPLY_REWRITE_PATTERN(TileWaitFlagToHIVM);
  APPLY_REWRITE_PATTERN(TilePipeBarrierToHIVM);
  APPLY_REWRITE_PATTERN(TileCubeWaitToHIVM);
  APPLY_REWRITE_PATTERN(TileGmOffsetToHIVM);

  // Step N: After all tile ops are lowered, convert !tile.buf types remaining
  // in tt.func signatures and tt.call ops (these arise when tile.alloc results
  // are passed across function boundaries).
  module.walk([](triton::FuncOp funcOp) {
    auto funcType = funcOp.getFunctionType();
    bool changed = false;

    // Convert input types
    SmallVector<Type> newInputs;
    for (auto ty : funcType.getInputs()) {
      if (auto bufTy = dyn_cast<tile::BufType>(ty)) {
        newInputs.push_back(convertBufToMemRef(bufTy));
        changed = true;
      } else {
        newInputs.push_back(ty);
      }
    }

    // Convert result types
    SmallVector<Type> newResults;
    for (auto ty : funcType.getResults()) {
      if (auto bufTy = dyn_cast<tile::BufType>(ty)) {
        newResults.push_back(convertBufToMemRef(bufTy));
        changed = true;
      } else {
        newResults.push_back(ty);
      }
    }

    if (!changed)
      return;

    // Update function type
    auto newFuncType =
        FunctionType::get(funcOp.getContext(), newInputs, newResults);
    funcOp.setFunctionType(newFuncType);

    // Update block argument types
    if (!funcOp.empty()) {
      Block &entry = funcOp.front();
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (auto bufTy =
                dyn_cast<tile::BufType>(entry.getArgument(i).getType())) {
          entry.getArgument(i).setType(convertBufToMemRef(bufTy));
        }
      }
    }
  });

  // Update tt.call result types to match new function signatures.
  module.walk([](triton::CallOp callOp) {
    for (unsigned i = 0; i < callOp->getNumResults(); ++i) {
      if (auto bufTy =
              dyn_cast<tile::BufType>(callOp->getResult(i).getType())) {
        callOp->getResult(i).setType(convertBufToMemRef(bufTy));
      }
    }
  });

  APPLY_REWRITE_PATTERN(LowerPtrTensorCastToMemref);
  APPLY_REWRITE_PATTERN(LowerMemrefCastToTensor);
  APPLY_REWRITE_PATTERN(FoldRoundtripCast);

  // Finally, fold any residual A<->B unrealized cast chains.
  SmallVector<UnrealizedConversionCastOp> casts;
  module->walk([&](UnrealizedConversionCastOp c) { casts.push_back(c); });
  reconcileUnrealizedCasts(casts);

  APPLY_REWRITE_PATTERN(FoldStagingCopyPattern);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCommonIRToHIVMPass() {
  return std::make_unique<CommonIRToHIVMPass>();
}
