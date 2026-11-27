//===------------------- TLEToMK.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/TLEToMK/TLEToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "tle-to-mk"

using namespace mlir;
using namespace triton;
using namespace mk;
using namespace tts;

namespace {

static constexpr llvm::StringLiteral kRemoteShardCarrierAttr =
    "tle.remote_shard_id_carrier";

static bool isConstantZeroIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (!isa<IndexType>(cst.getType()))
      return false;
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().isZero();
  }
  return false;
}

static bool areAllZeroIndices(ValueRange indices) {
  return llvm::all_of(indices, isConstantZeroIndex);
}

static bool isBeforeOrAtInSameBlock(Operation *a, Operation *b) {
  return a && b && a->getBlock() == b->getBlock() &&
         (a == b || a->isBeforeInBlock(b));
}

static Value getOrCreateScalarPtr(PatternRewriter &rewriter, Location loc,
                                  Value ptrLike, Operation *useAnchor) {
  if (!isa<RankedTensorType>(ptrLike.getType()))
    return ptrLike;

  for (Operation *user : ptrLike.getUsers()) {
    auto ex = dyn_cast<tensor::ExtractOp>(user);
    if (!ex)
      continue;
    if (ex.getTensor() != ptrLike)
      continue;
    if (!areAllZeroIndices(ex.getIndices()))
      continue;
    if (!useAnchor || isBeforeOrAtInSameBlock(ex.getOperation(), useAnchor))
      return ex.getResult();
  }

  auto ranked = cast<RankedTensorType>(ptrLike.getType());
  SmallVector<Value, 4> idxs;
  idxs.reserve(ranked.getRank());
  for (int i = 0; i < ranked.getRank(); ++i)
    idxs.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
  return tensor::ExtractOp::create(rewriter, loc, ptrLike, idxs);
}

static Value getOrCreatePtrToIntI64(PatternRewriter &rewriter, Location loc,
                                    Value scalarPtr, Operation *useAnchor) {
  for (Operation *user : scalarPtr.getUsers()) {
    auto p2i = dyn_cast<triton::PtrToIntOp>(user);
    if (!p2i)
      continue;
    if (p2i.getSrc() != scalarPtr)
      continue;
    if (p2i.getType() != rewriter.getI64Type())
      continue;
    if (!useAnchor || isBeforeOrAtInSameBlock(p2i.getOperation(), useAnchor))
      return p2i.getResult();
  }

  return triton::PtrToIntOp::create(rewriter, loc, rewriter.getI64Type(),
                                    scalarPtr);
}

/// Extract a flat i64 base-address from a pointer-like value.
///
/// When \p ptrLike is the result of a \c dsa.local_pointers op we go straight
/// to the underlying memref, avoiding the creation of any \c !tt.ptr typed
/// intermediate values.
static Value getOrCreatePtrLikeAddrI64(PatternRewriter &rewriter, Location loc,
                                       Value ptrLike, Operation *useAnchor) {
  // --- Fast path: dsa.local_pointers → extract base from memref directly ---
  if (auto localPtrOp = ptrLike.getDefiningOp<mlir::dsa::LocalPointersOp>()) {
    OpBuilder::InsertionGuard g(rewriter);
    // Insert right after the local_pointers op so that the new ops dominate
    // all users.
    if (localPtrOp->getNextNode())
      rewriter.setInsertionPoint(localPtrOp->getNextNode());
    else
      rewriter.setInsertionPointAfter(localPtrOp);
    auto idxTy = rewriter.getIndexType();
    auto i64Ty = rewriter.getI64Type();
    Value baseIndex = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, idxTy, localPtrOp.getSrc());
    return arith::IndexCastOp::create(rewriter, loc, i64Ty, baseIndex);
  }

  // --- Original path: Triton pointer value ---
  OpBuilder::InsertionGuard g(rewriter);
  Block *block = rewriter.getInsertionBlock();
  if (auto def = ptrLike.getDefiningOp()) {
    if (block && def->getBlock() == block)
      rewriter.setInsertionPointAfter(def);
  } else if (block) {
    rewriter.setInsertionPointToStart(block);
  }

  Value scalarPtr = getOrCreateScalarPtr(rewriter, loc, ptrLike, useAnchor);
  return getOrCreatePtrToIntI64(rewriter, loc, scalarPtr, useAnchor);
}

static Value castIntegerLikeToI64(PatternRewriter &rewriter, Location loc,
                                  Value v) {
  auto i64Ty = rewriter.getI64Type();
  Type ty = v.getType();
  if (ty == i64Ty)
    return v;
  if (isa<IndexType>(ty))
    return arith::IndexCastOp::create(rewriter, loc, i64Ty, v);
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() < 64)
      return arith::ExtSIOp::create(rewriter, loc, i64Ty, v);
    if (intTy.getWidth() > 64)
      return arith::TruncIOp::create(rewriter, loc, i64Ty, v);
    return v;
  }
  return Value();
}

static Value peelShardScalar(Value shardLike) {
  if (auto splat = shardLike.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  return shardLike;
}

static LogicalResult getCoordsFromShardIdValue(PatternRewriter &rewriter,
                                               Location loc, Value shardIdLike,
                                               SmallVector<Value, 4> &coords) {
  Value shardId = peelShardScalar(shardIdLike);
  Value tileId = castIntegerLikeToI64(rewriter, loc, shardId);
  if (!tileId)
    return failure();
  // The shardId passed from remote(buf, target_shard_id) is already a
  // physical tile ID (pre-computed by the user kernel from the mesh
  // topology LUT).  Pass it through directly — no modulo/division
  // decomposition into a fake 2D chip mesh.
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));
  coords = {zero, zero, zero, tileId};
  return success();
}

static LogicalResult
extractRemoteInfoFromPtr(PatternRewriter &rewriter, Location loc, Value ptrLike,
                         SmallVector<Value, 4> &coords, Value &basePtrLike,
                         DenseI32ArrayAttr *meshPhysicalIdsOut = nullptr,
                         DenseI32ArrayAttr *meshShapeOut = nullptr) {
  if (auto remotePtrOp = ptrLike.getDefiningOp<mlir::dsa::RemotePointersOp>()) {
    if (failed(getCoordsFromShardIdValue(rewriter, loc,
                                         remotePtrOp.getShardId(), coords)))
      return failure();
    basePtrLike = remotePtrOp.getSrc();
    if (meshPhysicalIdsOut)
      *meshPhysicalIdsOut = remotePtrOp.getMeshPhysicalIdsAttr();
    if (meshShapeOut)
      *meshShapeOut = remotePtrOp.getMeshShapeAttr();
    return success();
  }
  if (auto addPtr = ptrLike.getDefiningOp<triton::AddPtrOp>();
      addPtr && addPtr->hasAttr(kRemoteShardCarrierAttr)) {
    if (failed(getCoordsFromShardIdValue(rewriter, loc, addPtr.getOffset(),
                                         coords)))
      return failure();
    basePtrLike = addPtr.getPtr();
    return success();
  }
  return failure();
}

// ===----------------------------------------------------------------------===//
// Barrier
// ===----------------------------------------------------------------------===//

struct DsaDistributedBarrierToMkPattern
    : public OpRewritePattern<mlir::dsa::DistributedBarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::dsa::DistributedBarrierOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    DenseI32ArrayAttr meshPhysicalIds = op.getGroupMaskAttr();
    DenseI32ArrayAttr meshShape = op.getGroupShapeAttr();

    if (meshPhysicalIds && meshShape && !meshPhysicalIds.asArrayRef().empty()) {
      // Subgroup barrier: carry mesh topology through the pipeline via a
      // dedicated DistributeBarrierOp, leaving the plain BarrierOp untouched.
      mlir::mk::DistributeBarrierOp::create(rewriter, loc, meshPhysicalIds,
                                            meshShape);
    } else {
      // No group attributes → plain full-cluster barrier.
      mlir::mk::BarrierOp::create(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// Remote load / store  (dsa.remote_pointers → mk.remote_load/store)
// ===----------------------------------------------------------------------===//

struct DsaRemoteLoadToMkPattern : public OpRewritePattern<triton::LoadOp> {
  explicit DsaRemoteLoadToMkPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::LoadOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    SmallVector<Value, 4> recvCoords;
    Value basePtrLike = loadOp.getPtr();
    if (failed(extractRemoteInfoFromPtr(rewriter, loc, loadOp.getPtr(),
                                        recvCoords, basePtrLike)))
      return failure();

    auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    if (!resultType)
      return loadOp->emitRemark(
          "remote load currently expects ranked tensor result");
    for (int64_t s : resultType.getShape()) {
      if (ShapedType::isDynamic(s))
        return loadOp->emitRemark(
            "remote load with dynamic shape not supported");
    }

    Value dstBuffer = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());
    auto recvOp = mk::RemoteLoadOp::create(
        rewriter, loc, resultType, recvCoords[0], recvCoords[1], recvCoords[2],
        recvCoords[3], dstBuffer);
    rewriter.replaceOp(loadOp, recvOp.getResults().front());
    return success();
  }
};

struct DsaRemoteStoreToMkPattern : public OpRewritePattern<triton::StoreOp> {
  explicit DsaRemoteStoreToMkPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::StoreOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(triton::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    SmallVector<Value, 4> sendCoords;
    Value basePtrLike = storeOp.getPtr();
    DenseI32ArrayAttr meshPhysicalIds;
    DenseI32ArrayAttr meshShape;
    if (failed(extractRemoteInfoFromPtr(rewriter, loc, storeOp.getPtr(),
                                        sendCoords, basePtrLike,
                                        &meshPhysicalIds, &meshShape)))
      return failure();

    if (storeOp.getMask())
      return storeOp->emitRemark("masked remote store not supported");

    Value dstAddrI64 = getOrCreatePtrLikeAddrI64(rewriter, loc, basePtrLike,
                                                 storeOp.getOperation());
    mk::RemoteStoreOp::create(rewriter, loc, sendCoords[0], sendCoords[1],
                              sendCoords[2], sendCoords[3], dstAddrI64,
                              storeOp.getValue(), meshPhysicalIds, meshShape);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// Local load / store  (dsa.local_pointers + tt.load/store → memref ops)
//
// Instead of lowering dsa.local_pointers to Triton pointer arithmetic
// (tt.splat/tt.addptr with tensor<!tt.ptr<...>>), we directly convert the
// load/store users to memref-level operations.  This avoids producing
// !tt.ptr element types that downstream triton-to-core-dialects cannot
// convert to valid memref types.
// ===----------------------------------------------------------------------===//

/// tt.load whose pointer comes from dsa.local_pointers →
///     bufferization.to_tensor of the underlying memref.
struct DsaLocalLoadToMemrefPattern : public OpRewritePattern<triton::LoadOp> {
  explicit DsaLocalLoadToMemrefPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::LoadOp>(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Only match loads whose pointer is produced by dsa.local_pointers.
    auto localPtrOp =
        loadOp.getPtr().getDefiningOp<mlir::dsa::LocalPointersOp>();
    if (!localPtrOp)
      return failure();

    auto memrefTy = dyn_cast<MemRefType>(localPtrOp.getSrc().getType());
    if (!memrefTy)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    if (!resultTy)
      return failure();

    // Build a tensor type from the memref shape + element type.
    auto tensorTy =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());

    // Shapes must agree (the common DSA pattern uses identity indices).
    if (tensorTy.getShape() != resultTy.getShape())
      return loadOp->emitRemark(
          "local load shape mismatch between memref and result tensor");

    // Element type may differ if an implicit cast is present (e.g. f32→f16).
    // For now we require them to match.
    if (memrefTy.getElementType() != resultTy.getElementType())
      return loadOp->emitRemark(
          "local load element type mismatch between memref and result tensor");

    // Replace with: bufferization.to_tensor %memref
    // writable=true because the SPM buffer is mutable.
    auto toTensor = bufferization::ToTensorOp::create(
        rewriter, loadOp.getLoc(), tensorTy, localPtrOp.getSrc(),
        /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(loadOp, toTensor.getResult());
    return success();
  }
};

/// tt.store whose pointer comes from dsa.local_pointers →
///     bufferization.to_buffer + memref.copy into the underlying SPM buffer.
struct DsaLocalStoreToMemrefPattern : public OpRewritePattern<triton::StoreOp> {
  explicit DsaLocalStoreToMemrefPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::StoreOp>(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(triton::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto localPtrOp =
        storeOp.getPtr().getDefiningOp<mlir::dsa::LocalPointersOp>();
    if (!localPtrOp)
      return failure();

    auto destMemrefTy = dyn_cast<MemRefType>(localPtrOp.getSrc().getType());
    if (!destMemrefTy)
      return failure();

    Value val = storeOp.getValue();
    auto valTy = dyn_cast<RankedTensorType>(val.getType());
    if (!valTy)
      return failure();

    // Shapes must match.
    if (valTy.getShape() != destMemrefTy.getShape())
      return storeOp->emitRemark(
          "local store shape mismatch between value tensor and SPM memref");

    // Element types must match (no implicit cast support yet).
    if (valTy.getElementType() != destMemrefTy.getElementType())
      return storeOp->emitRemark(
          "local store element type mismatch between value and SPM memref");

    Location loc = storeOp.getLoc();

    // Materialise the tensor value as a memref, then copy into the SPM buffer.
    // Use a contiguous memref type for the intermediate to_buffer result.
    auto srcMemrefTy =
        MemRefType::get(valTy.getShape(), valTy.getElementType());
    auto srcMemref =
        bufferization::ToBufferOp::create(rewriter, loc, srcMemrefTy, val);
    memref::CopyOp::create(rewriter, loc, srcMemref, localPtrOp.getSrc());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// dsa.randgen → mk.randgen
// ===----------------------------------------------------------------------===//

struct DsaRandGenToMkPattern : public OpRewritePattern<mlir::dsa::RandGenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::dsa::RandGenOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto seed0Ty = dyn_cast<RankedTensorType>(op.getSeed0().getType());
    auto seed1Ty = dyn_cast<RankedTensorType>(op.getSeed1().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getOut().getType());
    auto seed0OutTy = dyn_cast<RankedTensorType>(op.getSeed0Out().getType());
    auto seed1OutTy = dyn_cast<RankedTensorType>(op.getSeed1Out().getType());
    if (!seed0Ty || !seed1Ty || !outTy || !seed0OutTy || !seed1OutTy)
      return rewriter.notifyMatchFailure(
          op, "dsa.randgen expects ranked tensor operands/results");
    if (seed0Ty.getShape() != ArrayRef<int64_t>({16}) ||
        seed1Ty.getShape() != ArrayRef<int64_t>({16}) ||
        seed0OutTy.getShape() != ArrayRef<int64_t>({16}) ||
        seed1OutTy.getShape() != ArrayRef<int64_t>({16}))
      return rewriter.notifyMatchFailure(
          op, "dsa.randgen seeds must have shape [16]");
    if (!seed0Ty.getElementType().isInteger(64) ||
        !seed1Ty.getElementType().isInteger(64) ||
        !outTy.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(
          op, "dsa.randgen currently supports only i64 element type");

    int32_t byteCount = op.getByteCount();
    if (byteCount <= 0 || (byteCount % 128) != 0)
      return rewriter.notifyMatchFailure(
          op, "dsa.randgen byte_count must be a positive multiple of 128");
    int64_t expectedOutElems = static_cast<int64_t>(byteCount) / 8;
    if (outTy.getNumElements() != expectedOutElems)
      return rewriter.notifyMatchFailure(
          op, "dsa.randgen out numel must equal byte_count / 8");

    auto outInit = tensor::EmptyOp::create(rewriter, loc, outTy.getShape(),
                                           outTy.getElementType());
    auto seed0Init = tensor::EmptyOp::create(
        rewriter, loc, seed0OutTy.getShape(), seed0OutTy.getElementType());
    auto seed1Init = tensor::EmptyOp::create(
        rewriter, loc, seed1OutTy.getShape(), seed1OutTy.getElementType());

    auto mkOp = mk::RandGenOp::create(
        rewriter, loc, TypeRange{outTy, seed0OutTy, seed1OutTy}, op.getSeed0(),
        op.getSeed1(), outInit, seed0Init, seed1Init,
        rewriter.getI32IntegerAttr(byteCount), op.getFmtAttr());

    rewriter.replaceOp(op, mkOp->getResults());
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// dsa.bitcast → mk.bitcast (zero-cost SPM buffer alias)
// ===----------------------------------------------------------------------===//

struct DsaBitcastToMkPattern : public OpRewritePattern<mlir::dsa::BitcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::dsa::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mk::BitcastOp>(op, op.getResult().getType(),
                                               op.getSrc());
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// Remote pointers fallback (kept for edge cases)
// ===----------------------------------------------------------------------===//

struct DsaRemotePointersToTritonPattern
    : public OpRewritePattern<mlir::dsa::RemotePointersOp> {
  explicit DsaRemotePointersToTritonPattern(MLIRContext *ctx)
      : OpRewritePattern<mlir::dsa::RemotePointersOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(mlir::dsa::RemotePointersOp op,
                                PatternRewriter &rewriter) const override {
    Value offset = op.getShardId();
    if (auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType())) {
      auto shardTy = dyn_cast<RankedTensorType>(offset.getType());
      if (!shardTy || shardTy.getShape() != srcTy.getShape()) {
        auto offsetTy =
            RankedTensorType::get(srcTy.getShape(), offset.getType());
        offset =
            triton::SplatOp::create(rewriter, op.getLoc(), offsetTy, offset);
      }
    }
    auto addPtr = triton::AddPtrOp::create(rewriter, op.getLoc(), op.getType(),
                                           op.getSrc(), offset);
    addPtr->setAttr(kRemoteShardCarrierAttr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, addPtr.getResult());
    return success();
  }
};

// dsa.extract_slice / dsa.insert_slice -> tensor.extract_slice/insert_slice.
// `static_offsets` uses ShapedType::kDynamic as a sentinel for positions whose
// runtime value is carried by the variadic `offsets` operands.

// Rebuild mixed static/dynamic per-dim offsets into OpFoldResult list.
// Static positions fold to i64 attributes; dynamic ones are 0-d tensors or
// scalar ints that are extracted (tensor.extract) and cast to index.
static LogicalResult buildSliceOffsets(PatternRewriter &rewriter, Location loc,
                                       ArrayRef<int64_t> staticOffsets,
                                       ValueRange dynOffsets,
                                       SmallVectorImpl<OpFoldResult> &offsets) {
  unsigned dynIdx = 0;
  for (int64_t s : staticOffsets) {
    if (ShapedType::isDynamic(s)) {
      if (dynIdx >= dynOffsets.size())
        return failure();
      Value v = dynOffsets[dynIdx++];
      if (isa<RankedTensorType>(v.getType()))
        v = tensor::ExtractOp::create(rewriter, loc, v, ValueRange{});
      if (!v.getType().isIndex())
        v = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                       v);
      offsets.push_back(v);
    } else {
      offsets.push_back(rewriter.getI64IntegerAttr(s));
    }
  }
  return success(dynIdx == dynOffsets.size());
}

static void buildSliceSizesStrides(PatternRewriter &rewriter,
                                   ArrayRef<int64_t> dims,
                                   SmallVectorImpl<OpFoldResult> &result) {
  for (int64_t d : dims)
    result.push_back(rewriter.getI64IntegerAttr(d));
}

struct DsaExtractSliceToTensorSlicePattern
    : public OpRewritePattern<mlir::dsa::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::dsa::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy)
      return failure();
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy)
      return failure();

    SmallVector<OpFoldResult> offsets;
    if (failed(buildSliceOffsets(rewriter, op.getLoc(), op.getStaticOffsets(),
                                 op.getOffsets(), offsets)))
      return failure();
    SmallVector<OpFoldResult> sizes, strides;
    buildSliceSizesStrides(rewriter, op.getSizes(), sizes);
    buildSliceSizesStrides(rewriter, op.getStrides(), strides);

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultTy, op.getSrc(), offsets, sizes, strides);
    return success();
  }
};

struct DsaInsertSliceToTensorSlicePattern
    : public OpRewritePattern<mlir::dsa::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::dsa::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy)
      return failure();

    SmallVector<OpFoldResult> offsets;
    if (failed(buildSliceOffsets(rewriter, op.getLoc(), op.getStaticOffsets(),
                                 op.getOffsets(), offsets)))
      return failure();
    SmallVector<OpFoldResult> sizes, strides;
    buildSliceSizesStrides(rewriter, op.getSizes(), sizes);
    buildSliceSizesStrides(rewriter, op.getStrides(), strides);

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.getTile(), op.getSrc(), offsets, sizes, strides);
    return success();
  }
};

// Three-operand elementwise arithmetic on memrefs; benefit=4 fires before
// DsaLocalLoadToMemrefPattern (benefit=3). MKToTx81 maps the arith op to
// tx.*VV.
template <typename DsaOpT, typename ArithOpT>
struct DsaBinaryOpToLinalgPattern : public OpRewritePattern<DsaOpT> {
  explicit DsaBinaryOpToLinalgPattern(MLIRContext *ctx)
      : OpRewritePattern<DsaOpT>(ctx, /*benefit=*/4) {}

  LogicalResult matchAndRewrite(DsaOpT op,
                                PatternRewriter &rewriter) const override {
    auto lhsTy = dyn_cast<MemRefType>(op.getLhs().getType());
    auto rhsTy = dyn_cast<MemRefType>(op.getRhs().getType());
    auto outTy = dyn_cast<MemRefType>(op.getOut().getType());
    if (!lhsTy || !rhsTy || !outTy)
      return failure();

    if (lhsTy.getShape() != rhsTy.getShape() ||
        lhsTy.getShape() != outTy.getShape())
      return op->emitRemark("dsa binary op shape mismatch between lhs/rhs/out");
    if (lhsTy.getElementType() != rhsTy.getElementType() ||
        lhsTy.getElementType() != outTy.getElementType())
      return op->emitRemark(
          "dsa binary op element type mismatch between lhs/rhs/out");

    Location loc = op.getLoc();
    auto elemTy = lhsTy.getElementType();
    auto rank = static_cast<int64_t>(lhsTy.getShape().size());
    auto identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap,
                                           identityMap};
    SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, mlir::utils::IteratorType::parallel);

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOut()}, indexingMaps, iteratorTypes);

    Block &block = linalgOp.getRegion().emplaceBlock();
    block.addArgument(elemTy, loc);
    block.addArgument(elemTy, loc);
    block.addArgument(elemTy, loc);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      Value result = ArithOpT::create(rewriter, loc, block.getArgument(0),
                                      block.getArgument(1));
      linalg::YieldOp::create(rewriter, loc, result);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// dsa.to_tensor / dsa.to_buffer → bufferization ops.

/// dsa.to_tensor %src {writable} : memref<...> -> tensor<...>
///     → bufferization.to_tensor %src {restrict, writable}
struct DsaToTensorToBufferizationPattern
    : public OpRewritePattern<mlir::dsa::ToTensorOp> {
  explicit DsaToTensorToBufferizationPattern(MLIRContext *ctx)
      : OpRewritePattern<mlir::dsa::ToTensorOp>(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(mlir::dsa::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto memrefTy = dyn_cast<MemRefType>(op.getSrc().getType());
    if (!memrefTy)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy)
      return failure();

    auto tensorTy =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());

    // Shapes must agree (identity view).
    if (tensorTy.getShape() != resultTy.getShape())
      return op->emitRemark(
          "dsa.to_tensor shape mismatch between memref and result tensor");

    // Element types must match (no implicit cast support yet).
    if (memrefTy.getElementType() != resultTy.getElementType())
      return op->emitRemark("dsa.to_tensor element type mismatch between "
                            "memref and result tensor");

    auto toTensor = bufferization::ToTensorOp::create(
        rewriter, op.getLoc(), tensorTy, op.getSrc(),
        /*restrict=*/true, /*writable=*/op.getWritable());
    rewriter.replaceOp(op, toTensor.getResult());
    return success();
  }
};

/// dsa.to_buffer %src, %dst : tensor<...>, memref<...>
///     → bufferization.to_buffer %src : memref<...>; memref.copy %tmp, %dst
struct DsaToBufferToBufferizationPattern
    : public OpRewritePattern<mlir::dsa::ToBufferOp> {
  explicit DsaToBufferToBufferizationPattern(MLIRContext *ctx)
      : OpRewritePattern<mlir::dsa::ToBufferOp>(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(mlir::dsa::ToBufferOp op,
                                PatternRewriter &rewriter) const override {
    Value val = op.getSrc();
    auto valTy = dyn_cast<RankedTensorType>(val.getType());
    if (!valTy)
      return failure();

    auto destMemrefTy = dyn_cast<MemRefType>(op.getDst().getType());
    if (!destMemrefTy)
      return failure();

    // Shapes must match.
    if (valTy.getShape() != destMemrefTy.getShape())
      return op->emitRemark(
          "dsa.to_buffer shape mismatch between value tensor and SPM memref");

    // Element types must match (no implicit cast support yet).
    if (valTy.getElementType() != destMemrefTy.getElementType())
      return op->emitRemark(
          "dsa.to_buffer element type mismatch between value and SPM memref");

    Location loc = op.getLoc();

    // Materialise the tensor value as a memref, then copy into the SPM buffer.
    auto srcMemrefTy =
        MemRefType::get(valTy.getShape(), valTy.getElementType());
    auto srcMemref =
        bufferization::ToBufferOp::create(rewriter, loc, srcMemrefTy, val);
    memref::CopyOp::create(rewriter, loc, srcMemref, op.getDst());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

// ===----------------------------------------------------------------------===//
// Shared TLE frontend ops (tle.*)
//
// The shared frontend in python/triton/experimental/tle emits backend-neutral
// tle.* ops for cumsum / extract_tile / insert_tile. Lowering them here keeps
// the vendor knowledge inside this backend instead of patching the shared
// Python layer.
// ===----------------------------------------------------------------------===//

namespace {

/// Decompose the linear tile id of a tle.extract_tile / tle.insert_tile op into
/// per-dim slice offsets.
///
/// The tile grid is `grid[i] = srcShape[i] / tileShape[i]` and the id is
/// row-major with the last dimension varying fastest, matching the shared
/// frontend's linearisation. Static ids fold to integer offsets; dynamic ones
/// become a div/rem/mul chain on the index operand, and the corresponding
/// positions are marked with ShapedType::kDynamic.
static LogicalResult decomposeTileIndex(PatternRewriter &rewriter, Location loc,
                                        Value index, ArrayRef<int64_t> srcShape,
                                        ArrayRef<int64_t> tileShape,
                                        SmallVectorImpl<int64_t> &staticOffsets,
                                        SmallVectorImpl<Value> &dynOffsets) {
  size_t rank = srcShape.size();
  if (rank == 0 || tileShape.size() != rank)
    return failure();

  SmallVector<int64_t> grid(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (tileShape[i] <= 0 || srcShape[i] % tileShape[i] != 0)
      return failure();
    grid[i] = srcShape[i] / tileShape[i];
  }

  Value idx = index;
  if (auto idxTy = dyn_cast<RankedTensorType>(idx.getType())) {
    if (idxTy.getRank() != 0)
      return failure();
    idx = tensor::ExtractOp::create(rewriter, loc, idx, ValueRange{});
  }
  if (!idx.getType().isInteger())
    return failure();

  // Static tile id: fold the grid decomposition at compile time.
  if (auto cst = idx.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      int64_t linear = intAttr.getInt();
      if (linear < 0)
        return failure();
      SmallVector<int64_t> coords(rank, 0);
      int64_t remain = linear;
      for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        coords[i] = remain % grid[i];
        remain /= grid[i];
      }
      if (remain != 0)
        return failure(); // tile id out of range
      for (size_t i = 0; i < rank; ++i)
        staticOffsets.push_back(coords[i] * tileShape[i]);
      return success();
    }
  }

  // Dynamic tile id: emit the same div/rem chain the shared frontend used to
  // build in Python.
  Type idxTy = idx.getType();
  SmallVector<Value> coords(rank);
  Value lin = idx;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    Value g = arith::ConstantIntOp::create(rewriter, loc, idxTy, grid[i]);
    coords[i] = arith::RemUIOp::create(rewriter, loc, lin, g);
    lin = arith::DivUIOp::create(rewriter, loc, lin, g);
  }
  for (size_t i = 0; i < rank; ++i) {
    Value t = arith::ConstantIntOp::create(rewriter, loc, idxTy, tileShape[i]);
    dynOffsets.push_back(arith::MulIOp::create(rewriter, loc, coords[i], t));
    staticOffsets.push_back(ShapedType::kDynamic);
  }
  return success();
}

static void buildUnitStrides(PatternRewriter &rewriter, size_t rank,
                             SmallVectorImpl<OpFoldResult> &strides) {
  for (size_t i = 0; i < rank; ++i)
    strides.push_back(rewriter.getI64IntegerAttr(1));
}

/// tle.exclusive_cumsum → mk.cumsum (same CRT entry as the removed dsa.cumsum).
struct TleExclusiveCumsumToMkPattern
    : public OpRewritePattern<tle::ExclusiveCumsumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tle::ExclusiveCumsumOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto exclusiveTy = dyn_cast<RankedTensorType>(op.getExclusive().getType());
    if (!inputTy || !exclusiveTy)
      return rewriter.notifyMatchFailure(
          op, "tle.exclusive_cumsum expects ranked tensor types");
    if (inputTy.getShape() != exclusiveTy.getShape() ||
        inputTy.getElementType() != exclusiveTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "tle.exclusive_cumsum src and exclusive result must match");

    SmallVector<int64_t> shape(inputTy.getShape().begin(),
                               inputTy.getShape().end());
    int64_t rank = static_cast<int64_t>(shape.size());
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (rank == 0 || axis != rank - 1)
      return rewriter.notifyMatchFailure(
          op, "tle.exclusive_cumsum supports only the last dimension");
    if (op.getReverse())
      return op->emitError("tle.exclusive_cumsum(reverse=True) is not "
                           "implemented for the tsingmicro backend");

    // The shared op carries no scratch pad size. The CRT scans each row with
    // `pad + extent` scratch slots and requires pad >= axis extent; 2 * extent
    // matches what the previous backend-private Python path used.
    int64_t pad = 2 * shape.back();

    auto exclusiveInit = tensor::EmptyOp::create(
        rewriter, loc, exclusiveTy.getShape(), exclusiveTy.getElementType());

    bool scalarTotal = !isa<RankedTensorType>(op.getTotal().getType());
    RankedTensorType totalBufferTy;
    if (scalarTotal)
      totalBufferTy = RankedTensorType::get({1}, inputTy.getElementType());
    else
      totalBufferTy = cast<RankedTensorType>(op.getTotal().getType());
    auto totalInit =
        tensor::EmptyOp::create(rewriter, loc, totalBufferTy.getShape(),
                                totalBufferTy.getElementType());

    SmallVector<int64_t> scratchShape(shape.begin(), shape.end());
    scratchShape.back() += pad;
    auto scratchTy =
        RankedTensorType::get(scratchShape, inputTy.getElementType());
    auto scratchInit = tensor::EmptyOp::create(
        rewriter, loc, scratchTy.getShape(), scratchTy.getElementType());

    auto mkOp = mk::CumsumOp::create(
        rewriter, loc, TypeRange{exclusiveTy, totalBufferTy, scratchTy},
        op.getSrc(), exclusiveInit, totalInit, scratchInit,
        rewriter.getI32IntegerAttr(axis), rewriter.getI64ArrayAttr(shape),
        rewriter.getI64IntegerAttr(pad));

    Value total = mkOp->getResult(1);
    if (scalarTotal) {
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      total = tensor::ExtractOp::create(rewriter, loc, total, ValueRange{zero});
    }

    rewriter.replaceOp(op, ValueRange{mkOp->getResult(0), total});
    return success();
  }
};

/// tle.extract_tile → tensor.extract_slice.
struct TleExtractTileToTensorSlicePattern
    : public OpRewritePattern<tle::ExtractTileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tle::ExtractTileOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    if (!srcTy || !resultTy || !tileShapeAttr)
      return rewriter.notifyMatchFailure(
          op, "tle.extract_tile needs ranked src/result and a tile_shape attr");

    ArrayRef<int64_t> tileShape = tileShapeAttr.asArrayRef();
    SmallVector<int64_t> staticOffsets;
    SmallVector<Value> dynOffsets;
    if (failed(decomposeTileIndex(rewriter, op.getLoc(), op.getIndex(),
                                  srcTy.getShape(), tileShape, staticOffsets,
                                  dynOffsets)))
      return rewriter.notifyMatchFailure(op, "cannot decompose the tile index");

    SmallVector<OpFoldResult> offsets;
    if (failed(buildSliceOffsets(rewriter, op.getLoc(), staticOffsets,
                                 dynOffsets, offsets)))
      return failure();
    SmallVector<OpFoldResult> sizes, strides;
    buildSliceSizesStrides(rewriter, tileShape, sizes);
    buildUnitStrides(rewriter, tileShape.size(), strides);

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultTy, op.getSrc(), offsets, sizes, strides);
    return success();
  }
};

/// tle.insert_tile → tensor.insert_slice.
struct TleInsertTileToTensorSlicePattern
    : public OpRewritePattern<tle::InsertTileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tle::InsertTileOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto tileTy = dyn_cast<RankedTensorType>(op.getTile().getType());
    if (!srcTy || !tileTy)
      return rewriter.notifyMatchFailure(
          op, "tle.insert_tile needs ranked src/tile types");

    ArrayRef<int64_t> tileShape = tileTy.getShape();
    SmallVector<int64_t> staticOffsets;
    SmallVector<Value> dynOffsets;
    if (failed(decomposeTileIndex(rewriter, op.getLoc(), op.getIndex(),
                                  srcTy.getShape(), tileShape, staticOffsets,
                                  dynOffsets)))
      return rewriter.notifyMatchFailure(op, "cannot decompose the tile index");

    SmallVector<OpFoldResult> offsets;
    if (failed(buildSliceOffsets(rewriter, op.getLoc(), staticOffsets,
                                 dynOffsets, offsets)))
      return failure();
    SmallVector<OpFoldResult> sizes, strides;
    buildSliceSizesStrides(rewriter, tileShape, sizes);
    buildUnitStrides(rewriter, tileShape.size(), strides);

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.getTile(), op.getSrc(), offsets, sizes, strides);
    return success();
  }
};

} // namespace

void mlir::triton::populateTLEToMKConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DsaRandGenToMkPattern, DsaBitcastToMkPattern>(
      patterns.getContext());

  // Benefit 4: dsa binary arithmetic (add/sub/mul/max/min) → linalg.generic.
  // Fires before DsaLocalLoadToMemrefPattern (benefit=3).
  patterns
      .add<DsaBinaryOpToLinalgPattern<mlir::dsa::AddOp, arith::AddFOp>,
           DsaBinaryOpToLinalgPattern<mlir::dsa::SubOp, arith::SubFOp>,
           DsaBinaryOpToLinalgPattern<mlir::dsa::MulOp, arith::MulFOp>,
           DsaBinaryOpToLinalgPattern<mlir::dsa::MaximumOp, arith::MaximumFOp>,
           DsaBinaryOpToLinalgPattern<mlir::dsa::MinimumOp, arith::MinimumFOp>,
           DsaBinaryOpToLinalgPattern<mlir::dsa::DivOp, arith::DivFOp>>(
          patterns.getContext());

  // Benefit 3: dsa.to_tensor / dsa.to_buffer direct lowering.
  patterns.add<DsaToTensorToBufferizationPattern,
               DsaToBufferToBufferizationPattern>(patterns.getContext());

  // Highest benefit (3): local load/store → memref ops.
  // These MUST fire before any pattern that would produce !tt.ptr types.
  patterns.add<DsaLocalLoadToMemrefPattern, DsaLocalStoreToMemrefPattern>(
      patterns.getContext());

  // Benefit 2: remote load/store → mk ops.
  patterns.add<DsaRemoteLoadToMkPattern, DsaRemoteStoreToMkPattern>(
      patterns.getContext());

  // Benefit 1: remaining remote_pointers / barrier.
  patterns
      .add<DsaRemotePointersToTritonPattern, DsaDistributedBarrierToMkPattern>(
          patterns.getContext());

  patterns.add<DsaExtractSliceToTensorSlicePattern,
               DsaInsertSliceToTensorSlicePattern>(patterns.getContext());

  // Shared TLE frontend ops. Lowering them here keeps the shared Python layer
  // free of backend-specific dispatch.
  patterns
      .add<TleExclusiveCumsumToMkPattern, TleExtractTileToTensorSlicePattern,
           TleInsertTileToTensorSlicePattern>(patterns.getContext());
}
