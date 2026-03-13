//===------------------- TLEToMK.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/TLEToMK/TLEToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    idxs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
  return rewriter.create<tensor::ExtractOp>(loc, ptrLike, idxs);
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

  return rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(),
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
    Value baseIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, idxTy, localPtrOp.getSrc());
    return rewriter.create<arith::IndexCastOp>(loc, i64Ty, baseIndex);
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
    return rewriter.create<arith::IndexCastOp>(loc, i64Ty, v);
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() < 64)
      return rewriter.create<arith::ExtSIOp>(loc, i64Ty, v);
    if (intTy.getWidth() > 64)
      return rewriter.create<arith::TruncIOp>(loc, i64Ty, v);
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
  Value four =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(4));
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  Value chipX = rewriter.create<arith::RemUIOp>(loc, tileId, four);
  Value chipY = rewriter.create<arith::DivUIOp>(loc, tileId, four);
  coords = {chipX, chipY, zero, tileId};
  return success();
}

static LogicalResult extractRemoteInfoFromPtr(PatternRewriter &rewriter,
                                              Location loc, Value ptrLike,
                                              SmallVector<Value, 4> &coords,
                                              Value &basePtrLike) {
  if (auto remotePtrOp = ptrLike.getDefiningOp<mlir::dsa::RemotePointersOp>()) {
    if (failed(getCoordsFromShardIdValue(rewriter, loc,
                                         remotePtrOp.getShardId(), coords)))
      return failure();
    basePtrLike = remotePtrOp.getSrc();
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
    rewriter.create<mlir::mk::BarrierOp>(op.getLoc());
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

    Value dstBuffer = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto recvOp = rewriter.create<mk::RemoteLoadOp>(
        loc, resultType, recvCoords[0], recvCoords[1], recvCoords[2],
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
    if (failed(extractRemoteInfoFromPtr(rewriter, loc, storeOp.getPtr(),
                                        sendCoords, basePtrLike)))
      return failure();

    if (storeOp.getMask())
      return storeOp->emitRemark("masked remote store not supported");

    Value dstAddrI64 = getOrCreatePtrLikeAddrI64(rewriter, loc, basePtrLike,
                                                 storeOp.getOperation());
    rewriter.create<mk::RemoteStoreOp>(loc, sendCoords[0], sendCoords[1],
                                       sendCoords[2], sendCoords[3], dstAddrI64,
                                       storeOp.getValue());
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
    auto toTensor = rewriter.create<bufferization::ToTensorOp>(
        loadOp.getLoc(), localPtrOp.getSrc(),
        /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(loadOp, toTensor.getResult());
    return success();
  }
};

/// tt.store whose pointer comes from dsa.local_pointers →
///     bufferization.to_memref + memref.copy into the underlying SPM buffer.
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
    // Use a contiguous memref type for the intermediate to_memref result.
    auto srcMemrefTy =
        MemRefType::get(valTy.getShape(), valTy.getElementType());
    auto srcMemref =
        rewriter.create<bufferization::ToMemrefOp>(loc, srcMemrefTy, val);
    rewriter.create<memref::CopyOp>(loc, srcMemref, localPtrOp.getSrc());
    rewriter.eraseOp(storeOp);
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
            rewriter.create<triton::SplatOp>(op.getLoc(), offsetTy, offset);
      }
    }
    auto addPtr = rewriter.create<triton::AddPtrOp>(op.getLoc(), op.getType(),
                                                    op.getSrc(), offset);
    addPtr->setAttr(kRemoteShardCarrierAttr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, addPtr.getResult());
    return success();
  }
};

} // namespace

void mlir::triton::populateTLEToMKConversionPatterns(
    RewritePatternSet &patterns) {
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
}
