/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <functional>
#include <map>
#include <numeric>
#include <utility>

#include "Analysis/FirstLastUserAnalysis.h"
#include "PatternTritonGPUOpToGCU.h"
#include "Utility.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "Dialect/GCUWS/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

// Lower ttg.local_alloc() without input to memref.alloc with address space 2.
// e.g. %2 = ttg.local_alloc : ()
//        -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
//  =>  %2 = memref.alloc() : memref<64x64xf32, 2>
struct TTSmemLocalAllocOpLowering
    : SharedConversionPattern<triton::gpu::LocalAllocOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle local_alloc without input (pure shared memory allocation)
    if (alloc.getSrc())
      return failure();

    enterTritionOp(rewriter, alloc.getOperation());
    if (pTagPool.isExistInMap(alloc.getOperation())) {
      pTagPool.releaseMap(alloc.getOperation());
    }

    auto memDescType = alloc.getType();
    auto elemType = memDescType.getElementType();
    auto shape = memDescType.getShape();

    auto memrefType = MemRefType::get(
        shape, elemType, AffineMap{},
        IntegerAttr::get(IntegerType::get(alloc.getContext(), 64),
                         /*addressSpace=*/2));

    Value output;
    auto waspAllocAttr = alloc->getAttrOfType<BoolAttr>("ttg.WASP_alloc");
    if (waspAllocAttr && waspAllocAttr.getValue()) {
      output = rewriter.create<memref::AllocOp>(alloc.getLoc(), memrefType);
    } else {
      auto loc = alloc.getLoc();
      auto lastUser =
          userAnalysis.getLastUser(alloc.getOperation()->getResults()[0]);
      output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                           replaced2Origin, memrefType);
    }
    if (Operation *defOp = output.getDefiningOp()) {
      if (auto meshShareAttr =
              alloc->getAttrOfType<BoolAttr>("ttg.local_alloc.mesh_share")) {
        defOp->setAttr("ttg.local_alloc.mesh_share", meshShareAttr);
      }
      if (auto meshAxisAttr =
              alloc->getAttrOfType<StringAttr>("ttg.local_alloc.mesh_axis")) {
        defOp->setAttr("ttg.local_alloc.mesh_axis", meshAxisAttr);
      }
    }

    leaveTritionOp(rewriter, alloc.getOperation());
    rewriter.replaceOp(alloc, output);
    return success();
  }
};

/// Write-back counterpart of stageViewToContiguous: copy a contiguous
/// buffer into a non-contiguous strided memref view (e.g. a 2D sub-tile or
/// a transposed view) with plain strided memref.store by the master
/// thread. Used by TTLocalStoreOpLowering when the store destination is a
/// non-contiguous MemDesc view, which the DTE DesliceStart path cannot
/// describe (DTE assumes a contiguous destination).
/// TODO(haizhu.shao): optimize the implementation.
static LogicalResult storeContiguousToView(PatternRewriter &rewriter,
                                           Location loc, Value contiguous,
                                           Value view) {
  auto srcTy = dyn_cast<MemRefType>(contiguous.getType());
  auto dstTy = dyn_cast<MemRefType>(view.getType());
  if (!srcTy || !dstTy || !srcTy.hasStaticShape() ||
      srcTy.getShape() != dstTy.getShape())
    return failure();

  int64_t rank = dstTy.getRank();
  int64_t total = dstTy.getNumElements();
  if (total <= 0)
    return failure();

  auto masterWarpId = getMasterThreadId(view.getParentRegion());
  auto isMasterThread = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto totalIdx = rewriter.create<arith::ConstantIndexOp>(loc, total);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  rewriter.create<scf::IfOp>(
      loc, isMasterThread, [&](OpBuilder &b, Location l) {
        auto loop = b.create<scf::ForOp>(l, zero, totalIdx, one);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());
          Value flat = loop.getInductionVar();
          // Delinearize flat into per-dim coordinates.
          SmallVector<Value> coords;
          Value rem = flat;
          for (int64_t d = rank - 1; d >= 0; --d) {
            auto dimCst =
                b.create<arith::ConstantIndexOp>(l, dstTy.getShape()[d]);
            coords.push_back(b.create<arith::RemSIOp>(l, rem, dimCst));
            rem = b.create<arith::DivSIOp>(l, rem, dimCst);
          }
          std::reverse(coords.begin(), coords.end());
          Value v = b.create<memref::LoadOp>(l, contiguous, coords);
          b.create<memref::StoreOp>(l, v, view, coords);
          // scf::ForOp::build already inserted the
          // body terminator; nothing else to add.
        }
        b.create<scf::YieldOp>(l);
      });

  // Make the stored data visible to all threads.
  rewriter.create<gpu::BarrierOp>(loc);
  return success();
}

/// Common handling for a MemDesc-typed operand whose (already lowered)
/// value may be a strided view (from TTMemDescSubslice/TransOpLowering)
/// wrapped in an UnrealizedConversionCastOp: look through the cast and
/// return the real value. Returns failure for non-contiguous strided
/// layouts, which the DTE paths cannot consume (they derive the pointer
/// via bufferPtr(), honoring the view offset, but never pass strides to
/// the engine).
static FailureOr<Value>
unwrapContiguousMemRef(Value v, ConversionPatternRewriter &rewriter,
                       Operation *op, const char *what) {
  Value real = stripUnrealizedConversionCastOp(v);
  auto ty = dyn_cast<MemRefType>(real.getType());
  if (!ty)
    return failure();
  if (!ty.getLayout().isIdentity() && !isContiguousRowMajor(ty))
    return rewriter.notifyMatchFailure(
        op, Twine(what) +
                ": non-contiguous strided view is not supported on the "
                "DTE path");
  return real;
}

// Lower triton_gcu.copy_global_to_local following the IsShareOutput path
// from GCULoadOpLowering. The output buffer comes from adaptor.getDstMem()
// (the already-lowered memref from TTSmemLocalAllocOpLowering).
struct TTSmemCopyGlobalToLocalOpLowering
    : SharedConversionPattern<triton::gcu::CopyGlobalToLocalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::CopyGlobalToLocalOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, loadOp.getOperation());
    if (pTagPool.isExistInMap(loadOp.getOperation())) {
      pTagPool.releaseMap(loadOp.getOperation());
    }

    auto loc = loadOp.getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // The dst memdesc may be a MemDesc view (subslice/trans) lowered to a
    // strided memref wrapped in an UnrealizedConversionCastOp. Consume the
    // real value. A contiguous strided view is passed through: the DTE
    // destination pointer derives from bufferPtr(), which honors the view
    // offset. A non-contiguous view cannot be a DTE destination.
    auto output = unwrapContiguousMemRef(adaptor.getDstMem(), rewriter,
                                         loadOp.getOperation(),
                                         "copy_global_to_local dst");
    if (failed(output))
      return failure();
    auto outputType = dyn_cast<MemRefType>(output->getType());
    if (!outputType)
      return failure();

    Operation *firstUse = nullptr;
    Block *defBlock = loadOp->getBlock();
    for (Operation *user : loadOp.getDstMem().getUsers()) {
      if (user == loadOp.getOperation())
        continue;
      Operation *ancestor = user;
      while (ancestor && ancestor->getBlock() != defBlock)
        ancestor = ancestor->getParentOp();
      if (!ancestor)
        continue;
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }

    if (firstUse == nullptr &&
        loadOp->getParentOfType<gcu::WarpSpecializeOp>()) {
      for (Operation *op = loadOp->getNextNode(); op; op = op->getNextNode()) {
        if (isa<gcu::ProducerCommitOp, triton::gcuws::ProducerCommitOp>(op)) {
          firstUse = op;
          break;
        }
      }
    }

    auto elemType = outputType.getElementType();
    int64_t rank = outputType.getRank();

    auto asyncAttr =
        llvm::cast_if_present<BoolAttr>(loadOp->getAttr(kLoadAsync));
    triton::gcu::TagInfo tag;
    if (!asyncAttr || !asyncAttr.getValue() || firstUse == nullptr) {
      tag = pTagPool.getSharedSyncTagInfo(loadOp.getOperation());
    } else {
      tag = pTagPool.tryGetSharedAsyncTagInfo(loadOp.getOperation());
    }

    if (tag.isAsync()) {
      pTagPool.setMap(firstUse, tag);
    }

    auto defaultValue = loadOp.getDefaultValue()
        ? adaptor.getDefaultValue()
        : triton::gcu::createConstantZero(rewriter, loc, elemType);

    // Shape check: all shape dims > 0
    Value shapeCheck = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, adaptor.getShape()[0], zero);
    for (int64_t i = 1; i < rank; ++i) {
      auto dimCheck = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, adaptor.getShape()[i], zero);
      shapeCheck = rewriter.create<arith::AndIOp>(loc, shapeCheck, dimCheck);
    }

    // ConfigGcuLoad with IsShareOutput = true
    auto total_size = rewriter.create<scf::IfOp>(
        loc, shapeCheck,
        [&](OpBuilder builder, Location loc) {
          Value load_size = ConfigGcuLoad(
              builder, loc, *output,
              loadOp, outputType, adaptor.getPtr(),
              adaptor.getStrides(), adaptor.getShape(),
              defaultValue, tag, /*IsShareOutput=*/true,
              adaptor.getOffsets());
          builder.create<scf::YieldOp>(loc, ValueRange{load_size});
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::YieldOp>(loc, ValueRange{zero});
        }).getResult(0);

    auto ip = rewriter.saveInsertionPoint();
    if (tag.isAsync()) {
      rewriter.setInsertionPoint(firstUse);
    }

    // IsShareOutput wait: only thread0 waits, then barrier
    auto masterWarpId = getMasterThreadId(loadOp.getOperation());
    auto isMasterThread = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
        rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
    auto isAll =
        rewriter.create<arith::AndIOp>(loc, isMasterThread, shapeCheck);
    rewriter.create<scf::IfOp>(
        loc, isAll, [&](OpBuilder builder, Location loc) {
          WaitGcuLoadStore(builder, loc, tag, total_size);
          builder.create<scf::YieldOp>(loc);
        });
    if (!loadOp->getParentOfType<gcu::WarpSpecializeOp>() &&
        triton::gcu::getNumWarps(loadOp.getOperation()) > 1)
      rewriter.create<gpu::BarrierOp>(loc);
    rewriter.restoreInsertionPoint(ip);

    leaveTritionOp(rewriter, loadOp.getOperation());
    rewriter.eraseOp(loadOp);
    return success();
  }
};

// ============================================================================
// TTSmemGatherGlobalToLocalOpLowering
//
// Input classification:
//   1D / 2D-row-split  -> flat contiguous store  (emit1DGatherToLocal)
//   2D-col-split       -> per-row non-contiguous (emit2DColGatherToLocal)
//   3D+                -> error
//
// Read ops:
//   Full VL chunks -> tarGather (safe: always reads exactly VL from TAR)
//   Tail chunks    -> vector::GatherOp (tarGather would OOB on partial VL)
//
// Store ops:
//   Full VL without mask -> maskedstore (with all-true mask)
//   Partial / has mask   -> maskedstore (with createMask)
//
// Staged unroll: all reads first, then all stores.
// ============================================================================
struct TTSmemGatherGlobalToLocalOpLowering
    : SharedConversionPattern<triton::gcu::GatherGlobalToLocalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::GatherGlobalToLocalOp gatherOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, gatherOp.getOperation());
    if (pTagPool.isExistInMap(gatherOp.getOperation())) {
      pTagPool.releaseMap(gatherOp.getOperation());
    }

    if (cast<RankedTensorType>(gatherOp.getSrc().getType()).getRank() >= 3)
      return rewriter.notifyMatchFailure(
          gatherOp, "GatherGlobalToLocal 3D+ not yet supported");

    auto loc = gatherOp.getLoc();

    G2LEmitContext ctx;
    if (failed(buildContext(rewriter, loc, gatherOp, adaptor, ctx)))
      return failure();

    if (ctx.totalNumElems == 0) {
      leaveTritionOp(rewriter, gatherOp.getOperation());
      rewriter.eraseOp(gatherOp);
      return success();
    }

    if (ctx.isColSplit)
      emit2DColGatherToLocal(rewriter, loc, adaptor, ctx);
    else
      emit1DGatherToLocal(rewriter, loc, adaptor, ctx);

    if (!gatherOp->getParentOfType<gcu::WarpSpecializeOp>())
      rewriter.create<gpu::BarrierOp>(loc);

    leaveTritionOp(rewriter, gatherOp.getOperation());
    rewriter.eraseOp(gatherOp);
    return success();
  }

private:
  struct G2LEmitContext {
    Type elemType;
    unsigned bpe;
    unsigned VL;
    unsigned K;              // inner dim per warp
    unsigned M;              // outer dim per warp (1 for flat 1D)
    unsigned totalNumElems;
    unsigned totalCols;      // full smem row width
    bool isColSplit;

    Value flatDst;
    Value warpFlatBase;
    Value masksPtr;          // nullptr when no mask
    Value vtOther;           // passthru vector
    VectorType vtType;       // <VL x elemType>
    VectorType maskVecType;  // <VL x i1>
    Value totalColsVal;
    Value innerDimVal;       // ConstantIndexOp(K)

    // For vector::GatherOp tail path
    Value srcFlatI64;        // src ptrs as flat i64 memref
    Value baseI64;           // first ptr value (i64 scalar)
    Value baseMemref;        // base dynamic memref for GatherOp
    Value baseI64Vec;        // broadcast of baseI64 to <VL x i64>
    Value bpeVec;            // broadcast of bpe to <VL x i64>
    VectorType i64VecType;   // <VL x i64>
    VectorType i32VecType;   // <VL x i32>
    Value zeroI64Passthru;   // <VL x 0_i64>
  };

  LogicalResult buildContext(ConversionPatternRewriter &rewriter, Location loc,
                             triton::gcu::GatherGlobalToLocalOp gatherOp,
                             OpAdaptor adaptor, G2LEmitContext &ctx) const {
    // The dst memdesc may be a strided view (MemDesc subslice/trans
    // lowering) wrapped in an UnrealizedConversionCastOp. Consume the real
    // value; only identity or contiguous strided destinations can be
    // written by the gather lowering.
    auto dst = unwrapContiguousMemRef(adaptor.getDstMem(), rewriter,
                                      gatherOp.getOperation(),
                                      "gather_global_to_local dst");
    if (failed(dst))
      return failure();
    Value dstVal = *dst;
    auto dstMemref = dyn_cast<MemRefType>(dstVal.getType());
    if (!dstMemref)
      return failure();

    ctx.elemType = dstMemref.getElementType();
    ctx.bpe = triton::gcu::getBpe(ctx.elemType);
    ctx.VL = oaccSizeInBytes / ctx.bpe;

    auto srcPtrTensorType = cast<RankedTensorType>(gatherOp.getSrc().getType());
    auto pointeeType =
        cast<triton::PointerType>(srcPtrTensorType.getElementType())
            .getPointeeType();
    auto dataTensorType =
        RankedTensorType::get(srcPtrTensorType.getShape(), pointeeType,
                              srcPtrTensorType.getEncoding());

    auto numElems = triton::gcu::getElemsPerThread(dataTensorType);
    ctx.totalNumElems = triton::gcu::getTotalElemsPerThread(dataTensorType);
    unsigned rank = numElems.size();

    auto slicedAxies = getSlicedAxies(dataTensorType);
    ctx.isColSplit = (rank >= 2 && slicedAxies.contains(rank - 1));

    if (ctx.isColSplit) {
      ctx.M = numElems[0];
      ctx.K = numElems[rank - 1];
    } else {
      ctx.M = 1;
      ctx.K = ctx.totalNumElems;
    }

    auto dstShape = dstMemref.getShape();
    auto flatTotal = std::accumulate(dstShape.begin(), dstShape.end(), 1LL,
                                     std::multiplies<int64_t>());
    ctx.totalCols = dstShape[rank - 1];
    ctx.totalColsVal =
        rewriter.create<arith::ConstantIndexOp>(loc, ctx.totalCols);
    ctx.innerDimVal = rewriter.create<arith::ConstantIndexOp>(loc, ctx.K);

    auto flatType = MemRefType::get({flatTotal}, ctx.elemType, AffineMap{},
                                    dstMemref.getMemorySpace());
    // Fold the (possibly non-zero) view offset into the flat destination:
    // reinterpret_cast replaces the source metadata, so passing offset 0
    // on a strided view source would silently drop the view offset.
    auto [dstStrides, dstOffset] = dstMemref.getStridesAndOffset();
    (void)dstStrides;
    Value flatOffsetVal;
    if (ShapedType::isDynamic(dstOffset))
      flatOffsetVal =
          rewriter.create<gcu::GetMemRefOffsetOp>(loc, dstVal, ValueRange{});
    else
      flatOffsetVal = rewriter.create<arith::ConstantIndexOp>(loc, dstOffset);
    ctx.flatDst = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, dstVal, flatOffsetVal,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, flatTotal)},
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 1)});

    // Compute the base address in flattened local memory for each warp.
    //
    // Data layout: 2D tensor -> 1D linear address [row * totalCols + col]
    //
    // Row Split scenario:
    //   - Each warp handles different row ranges
    //   - Each warp occupies full rows (numElems[1] == totalCols)
    //   - warpFlatBase = warpId[0] * numElems[0] * totalCols
    //
    // Col Split scenario:
    //   - All warps handle different column ranges of the same row
    //   - Both row offset and column offset needed
    //   - warpFlatBase = warpId[0] * numElems[0] * totalCols +
    //                    warpId[1] * numElems[1]
    //
    // Example: 32x256, 4 warp, col split, each warp handles 32x64
    //   warp 0: base = 0*32*256 + 0*64 = 0    -> [0:32, 0:64]
    //   warp 1: base = 0*32*256 + 1*64 = 64   -> [0:32, 64:128]
    //   warp 2: base = 0*32*256 + 2*64 = 128  -> [0:32, 128:192]
    //   warp 3: base = 0*32*256 + 3*64 = 192  -> [0:32, 192:256]
    //
    // Example: 32x256, 4 warp, row split, each warp handles 8x256
    //   warp 0: base = 0*8*256 + 0*256 = 0    -> [0:8, 0:256]
    //   warp 1: base = 1*8*256 + 0*256 = 2048 -> [8:16, 0:256]
    //   warp 2: base = 2*8*256 + 0*256 = 4096 -> [16:24, 0:256]
    //   warp 3: base = 3*8*256 + 0*256 = 6144 -> [24:32, 0:256]
    auto warpIds = getWarpIds(rewriter, loc, dataTensorType);
    if (rank == 1) {
      ctx.warpFlatBase = rewriter.create<arith::MulIOp>(
          loc, warpIds[0],
          rewriter.create<arith::ConstantIndexOp>(loc, numElems[0]));
    } else {
      Value rowOff = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::MulIOp>(
              loc, warpIds[0],
              rewriter.create<arith::ConstantIndexOp>(loc, numElems[0])),
          ctx.totalColsVal);
      Value colOff = rewriter.create<arith::MulIOp>(
          loc, warpIds[rank - 1],
          rewriter.create<arith::ConstantIndexOp>(loc, numElems[rank - 1]));
      ctx.warpFlatBase = rewriter.create<arith::AddIOp>(loc, rowOff, colOff);
    }

    // Mask
    ctx.masksPtr = nullptr;
    if (adaptor.getMask()) {
      auto masksPtrType =
          gcu::PtrType::get(rewriter.getContext(), rewriter.getI1Type());
      ctx.masksPtr = rewriter.create<gcu::MemRefToPtrOp>(loc, masksPtrType,
                                                         adaptor.getMask());
    }

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Padding
    Value other;
    if (adaptor.getOther()) {
      auto otherVal = adaptor.getOther();
      if (auto mt = dyn_cast<MemRefType>(otherVal.getType())) {
        SmallVector<Value> idx(mt.getRank(), zero);
        other = rewriter.create<memref::LoadOp>(loc, otherVal, idx);
      } else {
        other = otherVal;
      }
    } else {
      other = triton::gcu::createConstantZero(rewriter, loc, ctx.elemType);
    }

    ctx.vtType = VectorType::get({static_cast<int64_t>(ctx.VL)}, ctx.elemType);
    ctx.maskVecType =
        VectorType::get({static_cast<int64_t>(ctx.VL)}, rewriter.getI1Type());
    ctx.vtOther = rewriter.create<vector::BroadcastOp>(loc, ctx.vtType, other);

    // Prepare vector::GatherOp infrastructure for tail handling.
    // The src memref contains i64 pointers; we need a base pointer and
    // offsets to use vector::GatherOp.
    const unsigned tailSize = ctx.K % ctx.VL;
    const bool needsTail = (tailSize != 0);
    if (needsTail) {
      auto srcMemrefType = dyn_cast<MemRefType>(adaptor.getSrc().getType());
      auto srcFlatSize = std::accumulate(srcMemrefType.getShape().begin(),
                                         srcMemrefType.getShape().end(), 1LL,
                                         std::multiplies<int64_t>());
      auto srcFlatI64Type =
          MemRefType::get({srcFlatSize}, rewriter.getI64Type(), AffineMap{},
                          srcMemrefType.getMemorySpace());
      ctx.srcFlatI64 = rewriter.create<memref::ReinterpretCastOp>(
          loc, srcFlatI64Type, adaptor.getSrc(), static_cast<int64_t>(0),
          ArrayRef<int64_t>{srcFlatSize}, ArrayRef<int64_t>{1LL});

      ctx.baseI64 = rewriter.create<memref::LoadOp>(loc, ctx.srcFlatI64,
                                                    ValueRange{zero});
      Value basePtr = rewriter.create<gcu::IntToPtrOp>(
          loc, gcu::PtrType::get(rewriter.getContext(), ctx.elemType),
          ctx.baseI64);
      auto baseDynMemrefType =
          MemRefType::get({ShapedType::kDynamic}, ctx.elemType);
      ctx.baseMemref =
          rewriter.create<gcu::PtrToMemRefOp>(loc, baseDynMemrefType, basePtr);

      ctx.i64VecType = VectorType::get({static_cast<int64_t>(ctx.VL)},
                                       rewriter.getI64Type());
      ctx.i32VecType = VectorType::get({static_cast<int64_t>(ctx.VL)},
                                       rewriter.getI32Type());
      ctx.baseI64Vec = rewriter.create<vector::BroadcastOp>(loc, ctx.i64VecType,
                                                            ctx.baseI64);
      Value bpeVal = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(ctx.bpe), 64);
      ctx.bpeVec =
          rewriter.create<vector::BroadcastOp>(loc, ctx.i64VecType, bpeVal);
      ctx.zeroI64Passthru = rewriter.create<vector::BroadcastOp>(
          loc, ctx.i64VecType,
          rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
    }

    return success();
  }

  static Value advanceMaskPtr(OpBuilder &builder, Location loc, Value maskPtr,
                              unsigned advance) {
    auto masksPtrType =
        gcu::PtrType::get(builder.getContext(), builder.getI1Type());
    return builder.create<gcu::IntToPtrOp>(
        loc, masksPtrType,
        builder.create<arith::AddIOp>(
            loc, builder.create<gcu::PtrToIntOp>(loc, maskPtr),
            builder.create<arith::ConstantIntOp>(
                loc, static_cast<int64_t>(advance), 64)));
  }

  // Emit a vector::GatherOp for tail (partial VL) reads.
  // Reads pointers from srcFlatI64 at the given index, computes offsets
  // relative to basePtr, then gathers data elements.
  Value emitVectorGather(OpBuilder &builder, Location loc,
                         const G2LEmitContext &ctx, Value srcPtrIdx,
                         Value gatherMask) const {
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value ptrVec = builder.create<vector::MaskedLoadOp>(
        loc, ctx.i64VecType, ctx.srcFlatI64, ValueRange{srcPtrIdx}, gatherMask,
        ctx.zeroI64Passthru);
    Value diffVec = builder.create<arith::SubIOp>(loc, ptrVec, ctx.baseI64Vec);
    // Use right shift instead of DivSI for bpe (always power of 2)
    unsigned bpeShift = llvm::Log2_32(ctx.bpe);
    Value shiftAmt = builder.create<arith::ConstantIntOp>(loc, bpeShift, 64);
    Value shiftVec =
        builder.create<vector::BroadcastOp>(loc, ctx.i64VecType, shiftAmt);
    Value offsetI64Vec = builder.create<arith::ShRSIOp>(loc, diffVec, shiftVec);
    Value offsetI32Vec =
        builder.create<arith::TruncIOp>(loc, ctx.i32VecType, offsetI64Vec);
    return builder.create<vector::GatherOp>(loc, ctx.vtType, ctx.baseMemref,
                                            ValueRange{zero}, offsetI32Vec,
                                            gatherMask, ctx.vtOther);
  }

  // ========================= 1D path (flat contiguous) =======================
  // Handles: pure 1D, and 2D-row-split (flattened to 1D).
  //
  // fullChunks: tarGather + maskedstore  (compile-time known count)
  // tailChunk:  vector::GatherOp + maskedstore (if tailSize != 0)
  //
  // Tail is handled inline: the last unrolled iteration uses vector::GatherOp
  // when we know at compile time that it corresponds to the tail.
  void emit1DGatherToLocal(ConversionPatternRewriter &rewriter, Location loc,
                           OpAdaptor adaptor, const G2LEmitContext &ctx) const {
    const unsigned fullChunks = ctx.K / ctx.VL;
    const unsigned tailSize = ctx.K % ctx.VL;
    const unsigned totalChunks = fullChunks + (tailSize > 0 ? 1 : 0);

    constexpr unsigned maxUnroll = 16;
    const unsigned unrollFactor = std::min(maxUnroll, totalChunks);
    const unsigned stepElems = unrollFactor * ctx.VL;
    const unsigned finalStep = std::min(stepElems, ctx.totalNumElems);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto end = rewriter.create<arith::ConstantIndexOp>(loc, ctx.totalNumElems);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, finalStep);

    triton::gcu::TritonGCUBuilder tarSetup(loc, rewriter);
    Value argsPtr = tarSetup.tarAddr(adaptor.getSrc());

    SmallVector<Value> initArgs = {argsPtr};
    if (ctx.masksPtr)
      initArgs.push_back(ctx.masksPtr);

    rewriter.create<scf::ForOp>(
        loc, zero, end, step, initArgs,
        [&](OpBuilder &builder, Location loc, Value loopIter,
            ValueRange iterArgs) {
          SmallVector<Value> args(iterArgs.begin(), iterArgs.end());
          triton::gcu::TritonGCUBuilder tb(loc, builder);

          // Phase 1: all reads
          SmallVector<Value> gathered;
          SmallVector<Value> actualNums;
          gathered.reserve(unrollFactor);
          actualNums.reserve(unrollFactor);

          for (unsigned u = 0; u < unrollFactor; ++u) {
            Value iter = builder.create<arith::AddIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc, u * ctx.VL));
            Value remaining = builder.create<arith::SubIOp>(loc, end, iter);
            Value actualNum = builder.create<arith::MinSIOp>(
                loc, builder.create<arith::ConstantIndexOp>(loc, ctx.VL),
                remaining);
            actualNums.push_back(actualNum);

            bool isTailSlot = (tailSize != 0) && (u == unrollFactor - 1);

            if (isTailSlot) {
              // Tail: use vector::GatherOp to avoid tarGather OOB
              Value tailMask = builder.create<vector::CreateMaskOp>(
                  loc, ctx.maskVecType, ValueRange{actualNum});
              Value v = emitVectorGather(builder, loc, ctx, iter, tailMask);
              gathered.push_back(v);
            } else {
              Value curMask = args.size() > 1 ? args[1] : ctx.masksPtr;
              Value numI32 = builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(), actualNum);
              Value v = tb.tarGather(ctx.vtType, args[0], numI32, ctx.vtOther,
                                     curMask);
              gathered.push_back(v);
            }

            if (args.size() > 1)
              args[1] = advanceMaskPtr(builder, loc, args[1], ctx.VL);
          }

          // Phase 2: all stores
          for (unsigned u = 0; u < unrollFactor; ++u) {
            Value iter = builder.create<arith::AddIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc, u * ctx.VL));
            Value flatOffset =
                builder.create<arith::AddIOp>(loc, ctx.warpFlatBase, iter);
            Value storeMask = builder.create<vector::CreateMaskOp>(
                loc, ctx.maskVecType, ValueRange{actualNums[u]});
            builder.create<vector::MaskedStoreOp>(loc, ctx.flatDst,
                                                  ValueRange{flatOffset},
                                                  storeMask, gathered[u]);
          }

          builder.create<scf::YieldOp>(loc, args);
        });
  }

  // =================== 2D col-split path (non-contiguous) ====================
  void emit2DColGatherToLocal(ConversionPatternRewriter &rewriter, Location loc,
                              OpAdaptor adaptor,
                              const G2LEmitContext &ctx) const {
    if (ctx.K >= ctx.VL)
      emit2DColLargeK(rewriter, loc, adaptor, ctx);
    else
      emit2DColSmallK(rewriter, loc, adaptor, ctx);
  }

  // 2D col-split, K >= VL: per-row processing, each row like 1D.
  // Unroll both M and K, staged pattern.
  // Avoids DivSI/RemSI by tracking row/col with compile-time increments.
  void emit2DColLargeK(ConversionPatternRewriter &rewriter, Location loc,
                       OpAdaptor adaptor, const G2LEmitContext &ctx) const {
    const unsigned chunksPerRow = (ctx.K + ctx.VL - 1) / ctx.VL;
    const unsigned tailSize = ctx.K % ctx.VL;

    constexpr unsigned maxUnroll = 16;
    const unsigned mUnroll = std::min(4u, ctx.M);
    const unsigned kUnroll = std::min(maxUnroll / mUnroll, chunksPerRow);
    const unsigned totalUnroll = mUnroll * kUnroll;
    const unsigned stepElems = totalUnroll * ctx.VL;
    const unsigned finalStep = std::min(stepElems, ctx.totalNumElems);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto end = rewriter.create<arith::ConstantIndexOp>(loc, ctx.totalNumElems);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, finalStep);

    triton::gcu::TritonGCUBuilder tarSetup(loc, rewriter);
    Value argsPtr = tarSetup.tarAddr(adaptor.getSrc());

    SmallVector<Value> initArgs = {argsPtr};
    if (ctx.masksPtr)
      initArgs.push_back(ctx.masksPtr);

    // Pre-compute per-unroll-slot row and column offsets at compile time
    // to avoid DivSI/RemSI inside the loop.
    struct SlotInfo {
      unsigned rowDelta;   // row offset from loop base row
      unsigned colOffset;  // column offset within the row
      bool isTail;         // whether this slot is a tail chunk
    };
    SmallVector<SlotInfo> slots;
    slots.reserve(totalUnroll);
    for (unsigned u = 0; u < totalUnroll; ++u) {
      unsigned elemOffset = u * ctx.VL;
      unsigned row = elemOffset / ctx.K;
      unsigned col = elemOffset % ctx.K;
      bool isTail = (tailSize != 0) && (col + ctx.VL > ctx.K);
      slots.push_back({row, col, isTail});
    }

    rewriter.create<scf::ForOp>(
        loc, zero, end, step, initArgs,
        [&](OpBuilder &builder, Location loc, Value loopIter,
            ValueRange iterArgs) {
          SmallVector<Value> args(iterArgs.begin(), iterArgs.end());
          triton::gcu::TritonGCUBuilder tb(loc, builder);

          // Compute base row from loopIter using multiply+shift instead
          // of DivSI. loopIter is always a multiple of stepElems so
          // baseRow = loopIter / K. Since K is compile-time const, we
          // use MulIOp + ShRUI when K is power-of-2, otherwise DivSI.
          Value baseRow;
          bool kIsPow2 = (ctx.K & (ctx.K - 1)) == 0 && ctx.K > 0;
          if (kIsPow2) {
            unsigned kShift = llvm::Log2_32(ctx.K);
            Value shiftVal =
                builder.create<arith::ConstantIndexOp>(loc, kShift);
            baseRow = builder.create<arith::ShRUIOp>(loc, loopIter, shiftVal);
          } else {
            baseRow =
                builder.create<arith::DivSIOp>(loc, loopIter, ctx.innerDimVal);
          }

          SmallVector<Value> gathered;
          SmallVector<Value> actualNums;
          SmallVector<Value> flatOffsets;
          gathered.reserve(totalUnroll);
          actualNums.reserve(totalUnroll);
          flatOffsets.reserve(totalUnroll);

          // Phase 1: all reads
          for (unsigned u = 0; u < totalUnroll; ++u) {
            Value iter = builder.create<arith::AddIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc, u * ctx.VL));
            Value remaining = builder.create<arith::SubIOp>(loc, end, iter);
            Value actualNum = builder.create<arith::MinSIOp>(
                loc, builder.create<arith::ConstantIndexOp>(loc, ctx.VL),
                remaining);
            actualNums.push_back(actualNum);

            // Row/col from compile-time slot info
            Value row = builder.create<arith::AddIOp>(
                loc, baseRow,
                builder.create<arith::ConstantIndexOp>(loc, slots[u].rowDelta));
            Value flatOff = builder.create<arith::AddIOp>(
                loc, ctx.warpFlatBase,
                builder.create<arith::AddIOp>(
                    loc,
                    builder.create<arith::MulIOp>(loc, row, ctx.totalColsVal),
                    builder.create<arith::ConstantIndexOp>(
                        loc, slots[u].colOffset)));
            flatOffsets.push_back(flatOff);

            if (slots[u].isTail) {
              Value tailMask = builder.create<vector::CreateMaskOp>(
                  loc, ctx.maskVecType, ValueRange{actualNum});
              Value v = emitVectorGather(builder, loc, ctx, iter, tailMask);
              gathered.push_back(v);
            } else {
              Value numI32 = builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(), actualNum);
              Value curMask = args.size() > 1 ? args[1] : ctx.masksPtr;
              Value v = tb.tarGather(ctx.vtType, args[0], numI32, ctx.vtOther,
                                     curMask);
              gathered.push_back(v);
            }

            if (args.size() > 1)
              args[1] = advanceMaskPtr(builder, loc, args[1], ctx.VL);
          }

          // Phase 2: all stores
          for (unsigned u = 0; u < totalUnroll; ++u) {
            Value storeMask = builder.create<vector::CreateMaskOp>(
                loc, ctx.maskVecType, ValueRange{actualNums[u]});
            builder.create<vector::MaskedStoreOp>(loc, ctx.flatDst,
                                                  ValueRange{flatOffsets[u]},
                                                  storeMask, gathered[u]);
          }

          builder.create<scf::YieldOp>(loc, args);
        });
  }

  // 2D col-split, K < VL: one tar_gather(VL) covers VL/K sub-rows.
  // Staged unroll on the M*K iteration space.
  // Row computation avoids DivSI by using compile-time known offsets
  // within each unrolled iteration, and a single MulIOp for the base.
  void emit2DColSmallK(ConversionPatternRewriter &rewriter, Location loc,
                       OpAdaptor adaptor, const G2LEmitContext &ctx) const {
    const unsigned numSubRows = ctx.VL / ctx.K;
    const unsigned tailLanes = ctx.VL % ctx.K;

    constexpr unsigned maxUnroll = 16;
    const unsigned unrollFactor =
        std::min(maxUnroll, (ctx.totalNumElems + ctx.VL - 1) / ctx.VL);
    const unsigned stepElems = unrollFactor * ctx.VL;
    const unsigned finalStep = std::min(stepElems, ctx.totalNumElems);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto end = rewriter.create<arith::ConstantIndexOp>(loc, ctx.totalNumElems);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, finalStep);

    // Pre-compute segment masks for sub-rows within a VL-wide gather
    const int64_t rowSkewStep = ctx.totalCols - ctx.K;
    SmallVector<Value> segMasks;
    SmallVector<int64_t> rowSkews;
    segMasks.reserve(numSubRows);
    rowSkews.reserve(numSubRows);

    for (unsigned r = 0; r < numSubRows; ++r) {
      rowSkews.push_back(r * rowSkewStep);
      SmallVector<bool> bits(ctx.VL, false);
      for (unsigned k = 0; k < ctx.K; ++k) {
        unsigned idx = r * ctx.K + k;
        if (idx < ctx.VL)
          bits[idx] = true;
      }
      segMasks.push_back(
          rewriter
              .create<arith::ConstantOp>(
                  loc,
                  DenseElementsAttr::get(ctx.maskVecType, llvm::ArrayRef(bits)))
              .getResult());
    }

    Value tailSegMask = nullptr;
    int64_t tailSkew = 0;
    if (tailLanes != 0) {
      SmallVector<bool> tailBits(ctx.VL, false);
      unsigned firstTail = numSubRows * ctx.K;
      for (unsigned k = 0; k < tailLanes; ++k)
        tailBits[firstTail + k] = true;
      tailSegMask = rewriter
                        .create<arith::ConstantOp>(
                            loc, DenseElementsAttr::get(
                                     ctx.maskVecType, llvm::ArrayRef(tailBits)))
                        .getResult();
      tailSkew = numSubRows * rowSkewStep;
    }

    triton::gcu::TritonGCUBuilder tarSetup(loc, rewriter);
    Value argsPtr = tarSetup.tarAddr(adaptor.getSrc());

    SmallVector<Value> initArgs = {argsPtr};
    if (ctx.masksPtr)
      initArgs.push_back(ctx.masksPtr);

    // Pre-compute the number of rows each unrolled gather covers.
    // gIter = g * VL elements into M*K space. Row = gIter / K.
    // Since VL and K are compile-time constants, row delta between
    // consecutive gathers is also constant: VL / K rows (+ possible extra).
    const unsigned rowsPerGather = ctx.VL / ctx.K;

    rewriter.create<scf::ForOp>(
        loc, zero, end, step, initArgs,
        [&](OpBuilder &builder, Location loc, Value loopIter,
            ValueRange iterArgs) {
          SmallVector<Value> args(iterArgs.begin(), iterArgs.end());
          triton::gcu::TritonGCUBuilder tb(loc, builder);

          SmallVector<Value> gathered;
          SmallVector<Value> predMasks;
          SmallVector<Value> rowBases;
          gathered.reserve(unrollFactor);
          predMasks.reserve(unrollFactor);
          rowBases.reserve(unrollFactor);

          Value curMask = args.size() > 1 ? args[1] : ctx.masksPtr;

          // Compute base row for loopIter.
          // loopIter / K, but K is compile-time known.
          Value baseRow;
          bool kIsPow2 = (ctx.K & (ctx.K - 1)) == 0 && ctx.K > 0;
          if (kIsPow2) {
            unsigned kShift = llvm::Log2_32(ctx.K);
            Value shiftVal =
                builder.create<arith::ConstantIndexOp>(loc, kShift);
            baseRow = builder.create<arith::ShRUIOp>(loc, loopIter, shiftVal);
          } else {
            baseRow =
                builder.create<arith::DivSIOp>(loc, loopIter, ctx.innerDimVal);
          }

          // Phase 1: all gathers
          for (unsigned g = 0; g < unrollFactor; ++g) {
            Value gIter = builder.create<arith::AddIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc, g * ctx.VL));

            Value gExceedsEnd = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sge, gIter, end);
            Value gValidNum = builder.create<arith::SelectOp>(
                loc, gExceedsEnd,
                builder.create<arith::ConstantIndexOp>(loc, 0),
                builder.create<arith::SubIOp>(loc, end, gIter));
            Value gActualNum = builder.create<arith::MinSIOp>(
                loc, builder.create<arith::ConstantIndexOp>(loc, ctx.VL),
                gValidNum);
            Value gPredMask = builder.create<vector::CreateMaskOp>(
                loc, ctx.maskVecType, ValueRange{gActualNum});

            // Row for this gather: baseRow + g * rowsPerGather
            unsigned gRowDelta = g * rowsPerGather;
            Value gRow = builder.create<arith::AddIOp>(
                loc, baseRow,
                builder.create<arith::ConstantIndexOp>(loc, gRowDelta));
            Value gBaseRowStart = builder.create<arith::AddIOp>(
                loc, ctx.warpFlatBase,
                builder.create<arith::MulIOp>(loc, gRow, ctx.totalColsVal));

            Value gNumI32 = builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(), gActualNum);
            Value v = tb.tarGather(ctx.vtType, args[0], gNumI32, ctx.vtOther,
                                   curMask);

            gathered.push_back(v);
            predMasks.push_back(gPredMask);
            rowBases.push_back(gBaseRowStart);

            if (curMask)
              curMask = advanceMaskPtr(builder, loc, curMask, ctx.VL);
          }

          if (args.size() > 1)
            args[1] = curMask;

          // Phase 2: all stores
          for (unsigned g = 0; g < unrollFactor; ++g) {
            for (unsigned r = 0; r < numSubRows; ++r) {
              Value storeMask =
                  builder.create<arith::AndIOp>(loc, segMasks[r], predMasks[g]);
              Value subOff = builder.create<arith::AddIOp>(
                  loc, rowBases[g],
                  builder.create<arith::ConstantIndexOp>(loc, rowSkews[r]));
              builder.create<vector::MaskedStoreOp>(
                  loc, ctx.flatDst, ValueRange{subOff}, storeMask, gathered[g]);
            }

            if (tailLanes != 0) {
              Value tailMask =
                  builder.create<arith::AndIOp>(loc, tailSegMask, predMasks[g]);
              Value tailOff = builder.create<arith::AddIOp>(
                  loc, rowBases[g],
                  builder.create<arith::ConstantIndexOp>(loc, tailSkew));
              builder.create<vector::MaskedStoreOp>(
                  loc, ctx.flatDst, ValueRange{tailOff}, tailMask, gathered[g]);
            }
          }

          builder.create<scf::YieldOp>(loc, args);
        });
  }
};

/// Copy a non-contiguous strided memref view into a freshly allocated
/// contiguous buffer of the same shape. The copy is performed by the master
/// thread with plain strided memref.load / memref.store (which honor
/// arbitrary strides), followed by a gpu.barrier so all threads observe the
/// staged data. This lets the local_load lowering paths (alias view / DTE
/// slice_start), which both assume a contiguous source, handle MemDesc view
/// sources such as 2D sub-tiles and transposed views correctly.
static FailureOr<Value> stageViewToContiguous(
    PatternRewriter &rewriter, Location loc, Value src,
    std::pair<Operation *, int> lastUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin) {
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy || !srcTy.hasStaticShape())
    return failure();

  auto elemTy = srcTy.getElementType();
  int64_t rank = srcTy.getRank();
  int64_t total = srcTy.getNumElements();
  if (total <= 0)
    return failure();

  // Staged buffer: same shape, identity layout, same memory space.
  auto stagedTy =
      MemRefType::get(srcTy.getShape(), elemTy, AffineMap{},
                      srcTy.getMemorySpace());
  Value staged = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                             replaced2Origin, stagedTy);

  auto masterWarpId = getMasterThreadId(src.getParentRegion());
  auto isMasterThread = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto totalIdx = rewriter.create<arith::ConstantIndexOp>(loc, total);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  rewriter.create<scf::IfOp>(loc, isMasterThread,
                             [&](OpBuilder &b, Location l) {
                               auto loop = b.create<scf::ForOp>(
                                   l, zero, totalIdx, one);
                               {
                                 OpBuilder::InsertionGuard guard(b);
                                 b.setInsertionPointToStart(loop.getBody());
                                 Value flat = loop.getInductionVar();
                                 // Delinearize flat into per-dim coordinates.
                                 SmallVector<Value> coords;
                                 Value rem = flat;
                                 for (int64_t d = rank - 1; d >= 0; --d) {
                                   auto dimCst =
                                       b.create<arith::ConstantIndexOp>(
                                           l, srcTy.getShape()[d]);
                                   coords.push_back(b.create<arith::RemSIOp>(
                                       l, rem, dimCst));
                                   rem = b.create<arith::DivSIOp>(l, rem,
                                                                  dimCst);
                                 }
                                 std::reverse(coords.begin(), coords.end());
                                 Value v = b.create<memref::LoadOp>(l, src,
                                                                    coords);
                                 b.create<memref::StoreOp>(l, v, staged,
                                                           coords);
                                 // scf::ForOp::build already inserted the
                                 // body terminator; nothing else to add.
                               }
                               b.create<scf::YieldOp>(l);
                             });

  // Make the staged data visible to all threads before the actual load.
  rewriter.create<gpu::BarrierOp>(loc);
  return staged;
}

struct TTLocalLoadOpLowering
    : SharedConversionPattern<triton::gpu::LocalLoadOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto srcLayout =
        cast<triton::gpu::TensorOrMemDesc>(op.getSrc().getType()).getEncoding();
    auto dstLayout = dyn_cast<RankedTensorType>(op.getType()).getEncoding();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto firstUser =
        userAnalysis.getFirstUser(op.getOperation()->getResults()[0]);
    triton::gcu::TagInfo tag;
    if (firstUser.first != nullptr) {
      tag = pTagPool.tryGetPrivateAsyncTagInfo(op);
    } else {
      tag = pTagPool.getPrivateSyncTagInfo(op);
    }
    if (tag.isAsync()) {
      pTagPool.setMap(firstUser.first, tag);
    }
    // share to Distributed
    if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      auto loc = op.getLoc();
      Value srcVal = stripUnrealizedConversionCastOp(adaptor.getSrc());
      auto srcMemRef = dyn_cast<MemRefType>(srcVal.getType());
      if (!srcMemRef)
        return failure();

      if (!srcMemRef.getLayout().isIdentity() &&
          !isContiguousRowMajor(srcMemRef)) {
        // Non-contiguous view source (e.g. a 2D sub-tile with a row stride
        // larger than its width, or a transposed view). Neither the alias
        // path (memref.view of a contiguous per-warp tile) nor the DTE
        // fallback (slice_start assumes a contiguous source) can describe
        // such a layout. Stage the view into a fresh contiguous buffer with
        // plain strided memref.load/store (which handle arbitrary strides),
        // then continue on the staged identity source.
        auto staged = stageViewToContiguous(rewriter, loc, srcVal, lastUser,
                                            userAnalysis, replaced2Origin);
        if (failed(staged))
          return rewriter.notifyMatchFailure(
              op, "local_load: failed to stage non-contiguous view source");
        srcVal = *staged;
        srcMemRef = cast<MemRefType>(srcVal.getType());
      }

      auto output = loadFromSharedMem(rewriter, tag, op.getResult().getType(),
                                      srcVal, false, lastUser, firstUser,
                                      userAnalysis, replaced2Origin, true);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      auto loc = op.getLoc();
      Value srcVal = stripUnrealizedConversionCastOp(adaptor.getSrc());
      auto srcMemRef = dyn_cast<MemRefType>(srcVal.getType());
      if (!srcMemRef)
        return failure();

      // The subview below derives its result strides from the source type;
      // a non-identity (MemDesc view) source would produce strides that do
      // not match the converter-provided dot-operand result type. Stage the
      // view into a contiguous buffer first, then slice as usual.
      if (!srcMemRef.getLayout().isIdentity()) {
        auto staged = stageViewToContiguous(rewriter, loc, srcVal, lastUser,
                                            userAnalysis, replaced2Origin);
        if (failed(staged))
          return rewriter.notifyMatchFailure(
              op, "local_load(dot_operand): failed to stage non-identity "
                  "source layout");
        srcVal = *staged;
      }

      auto numElems = triton::gcu::getElemsPerThread(op.getType());
      auto warpIds = getWarpIds(rewriter, loc, op.getType());
      unsigned rank = cast<MemRefType>(srcVal.getType()).getRank();

      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (unsigned i = 0; i < rank; ++i) {
        Value offset = rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::ConstantIndexOp>(loc, numElems[i]),
            warpIds[i]);
        offsets.push_back(offset);
        sizes.push_back(rewriter.getIndexAttr(numElems[i]));
        strides.push_back(rewriter.getIndexAttr(1));
      }

      auto resultType =
          cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
      auto subview = rewriter.create<memref::SubViewOp>(
          loc, resultType, srcVal, offsets, sizes, strides);

      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, subview.getResult());
      return success();
    } else {
      op.dump();
      llvm::report_fatal_error(
          "[Error] gcu::LocalLoadOp maybe had bad used in pinpong\n");
    }
    return success();
  }
};

// Lower ttg.local_store(src_tensor, dst_memdesc) via the storeToSharedMem
// utility which uses DTE DesliceStartOp for private->shared transfers.
struct TTLocalStoreOpLowering
    : SharedConversionPattern<triton::gpu::LocalStoreOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }

    auto srcTensorType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTensorType)
      return failure();

    if (!dyn_cast<MemRefType>(adaptor.getSrc().getType()))
      return failure();

    // The dst memdesc may be a MemDesc view (subslice/trans) lowered to a
    // strided memref wrapped in an UnrealizedConversionCastOp. Consume the
    // real value: for a contiguous view (e.g. a 1D sub-slice with offset
    // 0..N) DesliceStart is given the strided descriptor directly - the
    // DTE destination pointer derives from bufferPtr(), which honors the
    // view offset. A non-contiguous view (2D sub-tile, transpose) cannot
    // be a DTE destination: store to a contiguous staging buffer first,
    // then copy back into the view with strided memref.store.
    Value dstVal = stripUnrealizedConversionCastOp(adaptor.getDst());
    auto dstTy = dyn_cast<MemRefType>(dstVal.getType());
    if (!dstTy)
      return failure();

    auto tag = pTagPool.getPrivateSyncTagInfo(op);

    if (!dstTy.getLayout().isIdentity() && !isContiguousRowMajor(dstTy)) {
      // Non-contiguous view destination: stage through a contiguous buffer.
      auto loc = op.getLoc();
      auto stagedType =
          MemRefType::get(dstTy.getShape(), dstTy.getElementType(), AffineMap{},
                          dstTy.getMemorySpace());
      Value staged = syncAllocOp(rewriter, loc,
                                 std::make_pair(op.getOperation(), 0),
                                 userAnalysis, replaced2Origin, stagedType);
      storeToSharedMem(rewriter, tag, srcTensorType, staged,
                       adaptor.getSrc(), /*onlyThread0=*/false);
      if (failed(storeContiguousToView(rewriter, loc, staged, dstVal)))
        return rewriter.notifyMatchFailure(
            op, "local_store: failed to write back into the non-contiguous "
                "view destination");
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.eraseOp(op);
      return success();
    }

    storeToSharedMem(rewriter, tag, srcTensorType, dstVal,
                     adaptor.getSrc(), /*onlyThread0=*/false);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton_gcu.memdesc_to_ptr to gcu.memref2ptr.
// At this stage the type converter has already turned the !ttg.memdesc input
// into a memref and the !tt.ptr result into a gcu.ptr, so we simply extract
// the base address from the memref.
struct MemDescToPtrOpLowering
    : SharedConversionPattern<triton::gcu::MemDescToPtrOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::MemDescToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcVal = stripUnrealizedConversionCastOp(adaptor.getSrc());
    auto srcMemRef = dyn_cast<MemRefType>(srcVal.getType());
    if (!srcMemRef)
      return failure();

    auto ptr = getMemRefBasePtr(rewriter, srcVal);

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

// ============================================================================
// Helpers for extract_tile / insert_tile lowering
// ============================================================================

// Convert a linear tile index to multi-dim tile coordinates (row-major).
//
// The source tensor is viewed as a grid of tiles:
//   grid[d] = srcShape[d] / tileShape[d]
// Then linearIdx is delinearized into per-dim tile coordinates.
//   e.g. srcShape=[64,64], tileShape=[32,32] -> grid=[2,2]
//        linearIdx=3 -> coords=[1,1] (bottom-right tile)
static SmallVector<Value> delinearizeTileIndex(OpBuilder &b, Location loc,
                                               Value linearIdx,
                                               ArrayRef<int64_t> srcShape,
                                               ArrayRef<int64_t> tileShape) {
  unsigned rank = srcShape.size();
  SmallVector<int64_t> grid(rank);
  for (unsigned d = 0; d < rank; ++d)
    grid[d] = srcShape[d] / tileShape[d];

  auto i32Ty = b.getI32Type();
  Value idx = linearIdx;
  if (idx.getType() != i32Ty)
    idx = b.create<arith::IndexCastOp>(loc, i32Ty, idx);

  SmallVector<Value> coords(rank);
  for (unsigned d = 0; d < rank; ++d) {
    int64_t suffix = 1;
    for (unsigned dd = d + 1; dd < rank; ++dd)
      suffix *= grid[dd];

    if (suffix > 1) {
      auto suffixCst = b.create<arith::ConstantIntOp>(loc, suffix, 32);
      coords[d] = b.create<arith::DivSIOp>(loc, idx, suffixCst);
      idx = b.create<arith::RemSIOp>(loc, idx, suffixCst);
    } else {
      coords[d] = idx;
      idx = b.create<arith::ConstantIntOp>(loc, 0, 32);
    }
  }
  return coords;
}

// Convert a linear tile index to element-level offsets in the source tensor.
//
// First delinearizes the index into tile coordinates, then scales each
// coordinate by the tile size in that dimension to get the element offset.
//   e.g. srcShape=[64,64], tileShape=[32,32], linearIdx=3
//        -> coords=[1,1] -> offsets=[32,32]
//        meaning the tile starts at row 32, col 32 of the source.
static SmallVector<Value> computeElemOffsets(OpBuilder &b, Location loc,
                                             Value linearIdx,
                                             ArrayRef<int64_t> srcShape,
                                             ArrayRef<int64_t> tileShape) {
  auto coords = delinearizeTileIndex(b, loc, linearIdx, srcShape, tileShape);
  unsigned rank = srcShape.size();
  SmallVector<Value> offsets(rank);
  for (unsigned d = 0; d < rank; ++d) {
    auto cst = b.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(tileShape[d]), 32);
    offsets[d] = b.create<arith::MulIOp>(loc, coords[d], cst);
  }
  return offsets;
}

// Combine tile element offsets with per-warp offsets for SMEM relay.
//
// In the SMEM relay path, data is laid out in shared memory using the full
// source tensor shape. Each warp owns a contiguous slice along each dim,
// so the final read/write position is:
//   combinedOffsets[d] = elemOffsets[d] + numElems[d] * warpIds[d]
//
// where:
//   elemOffsets[d] - tile start in dim d        (from computeElemOffsets)
//   numElems[d]    - elements per warp in dim d (from getElemsPerThread)
//   warpIds[d]     - warp index in dim d        (from getWarpIds)
//
//   e.g. srcShape=[64,64], tileShape=[32,32], encoding warpsPerCTA=[1,2]
//        numElems=[32,16], warp 0: warpIds=[0,0], warp 1: warpIds=[0,1]
//        For tile index 0: elemOffsets=[0,0]
//          warp 0: combined=[0+32*0, 0+16*0]=[0,0]  -> reads SMEM[0:32, 0:16]
//          warp 1: combined=[0+32*0, 0+16*1]=[0,16] -> reads SMEM[0:32, 16:32]
static SmallVector<Value, 4> computeCombinedOffsets(OpBuilder &b, Location loc,
                                                    ArrayRef<Value> elemOffsets,
                                                    RankedTensorType tensorTy) {
  auto numElems = triton::gcu::getElemsPerThread(tensorTy);
  auto warpIds = getWarpIds(b, loc, tensorTy);
  unsigned rank = tensorTy.getRank();

  SmallVector<Value, 4> combined;
  for (unsigned d = 0; d < rank; ++d) {
    Value warpOff = b.create<arith::MulIOp>(
        loc, b.create<arith::ConstantIntOp>(loc, numElems[d], 32),
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), warpIds[d]));
    combined.push_back(b.create<arith::AddIOp>(loc, elemOffsets[d], warpOff));
  }
  return combined;
}

// ============================================================================
// Common SMEM relay lowering functions
//
// Shared by TleExtractTileLowering (fallback path) and
// SliceFromLocalOpLowering, and by TleInsertTileLowering (fallback path) and
// DesliceToLocalOpLowering.
// ============================================================================

// Emit SliceStartOp from SMEM with combined (tile + warp) offsets.
// smemBuf must already point to shared memory containing the source data.
static Value
emitSliceFromSmem(OpBuilder &rewriter, Location loc, Value smemBuf,
                  Value convertedIndex, ArrayRef<int64_t> srcShape,
                  ArrayRef<int64_t> tileShape, RankedTensorType resultTensorTy,
                  MemRefType outputType,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  triton::gcu::DTETagPool &pTagPool, Operation *op) {
  auto elemOffsets =
      computeElemOffsets(rewriter, loc, convertedIndex, srcShape, tileShape);
  auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);
  auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                            replaced2Origin, outputType);
  auto totalNumElems = triton::gcu::getTotalElemsPerThread(resultTensorTy);
  auto combinedOffsets =
      computeCombinedOffsets(rewriter, loc, elemOffsets, resultTensorTy);
  auto defaultValue = triton::gcu::createConstantZero(
      rewriter, loc, resultTensorTy.getElementType());

  auto sliceTag = pTagPool.getPrivateSyncTagInfo(op);
  rewriter.create<memref_ext::SliceStartOp>(
      loc, output, smemBuf, combinedOffsets, defaultValue, sliceTag.getTag(),
      ValueRange{sliceTag.getIdx()});
  rewriter.create<memref::DmaWaitOp>(
      loc, sliceTag.getTag(), ValueRange{sliceTag.getIdx()},
      rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
  return output;
}

// Emit DesliceStartOp into SMEM, barrier, then loadFromSharedMem.
// smemBuf must already point to shared memory containing the source data.
static Value
emitDesliceToSmem(OpBuilder &rewriter, Location loc, Value smemBuf,
                  Value convertedTile, Value convertedIndex,
                  ArrayRef<int64_t> srcShape, ArrayRef<int64_t> tileShape,
                  RankedTensorType tileTensorTy,
                  RankedTensorType resultTensorTy,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  triton::gcu::DTETagPool &pTagPool, Operation *op) {
  auto elemOffsets =
      computeElemOffsets(rewriter, loc, convertedIndex, srcShape, tileShape);
  auto firstUser = userAnalysis.getFirstUser(op->getResults()[0]);
  auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);

  auto totalNumElems = triton::gcu::getTotalElemsPerThread(tileTensorTy);
  auto combinedOffsets =
      computeCombinedOffsets(rewriter, loc, elemOffsets, tileTensorTy);

  auto desliceTag = pTagPool.getPrivateSyncTagInfo(op);
  rewriter.create<memref_ext::DesliceStartOp>(
      loc, smemBuf, convertedTile, combinedOffsets, desliceTag.getTag(),
      ValueRange{desliceTag.getIdx()});
  rewriter.create<memref::DmaWaitOp>(
      loc, desliceTag.getTag(), ValueRange{desliceTag.getIdx()},
      rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

  if (!op->getParentOfType<gcu::WarpSpecializeOp>())
    rewriter.create<gpu::BarrierOp>(loc);

  triton::gcu::TagInfo loadTag;
  if (firstUser.first != nullptr)
    loadTag = pTagPool.tryGetPrivateAsyncTagInfo(op);
  else
    loadTag = pTagPool.getPrivateSyncTagInfo(op);
  if (loadTag.isAsync())
    pTagPool.setMap(firstUser.first, loadTag);

  return loadFromSharedMem(rewriter, loadTag, resultTensorTy, smemBuf, false,
                           lastUser, firstUser, userAnalysis, replaced2Origin);
}

// ============================================================================
// Lowering: tle.extract_tile (tensor src)
//
// Private path: SliceStartOp with tile offsets only.
// SMEM fallback: storeToSharedMem + emitSliceFromSmem.
// ============================================================================
struct TleExtractTileLowering : SharedGenericConversionPattern {
  TleExtractTileLowering(const TypeConverter &converter, MLIRContext *ctx,
                         triton::gcu::FirstLastUserAnalysis &userAnalysis,
                         std::map<Operation *, Operation *> &replaced2Origin,
                         triton::gcu::DTETagPool &pTagPool)
      : SharedGenericConversionPattern("tle.extract_tile", converter, ctx,
                                       userAnalysis, replaced2Origin,
                                       pTagPool) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op))
      pTagPool.releaseMap(op);

    Value convertedSrc = operands[0];
    Value convertedIndex = operands[1];
    auto memType = dyn_cast<MemRefType>(convertedSrc.getType());
    if (!memType)
      return failure();

    auto loc = op->getLoc();
    auto resultTensorTy = cast<RankedTensorType>(op->getResult(0).getType());
    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    if (!tileShapeAttr)
      return failure();
    auto tileShape = tileShapeAttr.asArrayRef();

    auto srcTensorTy = cast<RankedTensorType>(op->getOperand(0).getType());
    auto srcShape = srcTensorTy.getShape();
    auto outputType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultTensorTy));

    if (!triton::gcu::needsSmemRelay(srcTensorTy, tileShape)) {
      auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, outputType);
      auto elemOffsets = computeElemOffsets(rewriter, loc, convertedIndex,
                                            srcShape, tileShape);
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(resultTensorTy);
      auto defaultValue = triton::gcu::createConstantZero(
          rewriter, loc, resultTensorTy.getElementType());
      auto sliceTag = pTagPool.getPrivateSyncTagInfo(op);
      rewriter.create<memref_ext::SliceStartOp>(
          loc, output, convertedSrc, elemOffsets, defaultValue,
          sliceTag.getTag(), ValueRange{sliceTag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, sliceTag.getTag(), ValueRange{sliceTag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

      leaveTritionOp(rewriter, op);
      rewriter.replaceOp(op, output);
      return success();
    }

    // SMEM relay path: check if shared memory is already allocated.
    auto memSpace = memType.getMemorySpace();
    bool isInSharedMem =
        (isa_and_nonnull<mlir::triton::gpu::SharedMemorySpaceAttr>(memSpace)) ||
        (isa_and_nonnull<IntegerAttr>(memSpace) &&
         cast<IntegerAttr>(memSpace).getInt() == 2);

    auto smemBuf = convertedSrc;
    if (!isInSharedMem) {
      // Shared memory not yet allocated (local-mem-optimize didn't fuse).
      // Store data to shared memory first.
      auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);
      auto storeTag = pTagPool.getPrivateSyncTagInfo(op);
      smemBuf = storeToSharedMem(rewriter, storeTag, srcTensorTy, convertedSrc,
                                 /*onlyThread0=*/false, lastUser, userAnalysis,
                                 replaced2Origin);
    }

    auto output =
        emitSliceFromSmem(rewriter, loc, smemBuf, convertedIndex, srcShape,
                          tileShape, resultTensorTy, outputType, userAnalysis,
                          replaced2Origin, pTagPool, op);

    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// ============================================================================
// Lowering: tle.insert_tile (tensor src)
//
// Private path: SliceStartOp (copy src) + DesliceStartOp (write tile).
// SMEM fallback: storeToSharedMem + emitDesliceToSmem.
// ============================================================================
struct TleInsertTileLowering : SharedGenericConversionPattern {
  TleInsertTileLowering(const TypeConverter &converter, MLIRContext *ctx,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin,
                        triton::gcu::DTETagPool &pTagPool)
      : SharedGenericConversionPattern("tle.insert_tile", converter, ctx,
                                       userAnalysis, replaced2Origin,
                                       pTagPool) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();

    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op))
      pTagPool.releaseMap(op);

    Value convertedSrc = operands[0];
    Value convertedTile = operands[1];
    Value convertedIndex = operands[2];
    auto memType = dyn_cast<MemRefType>(convertedSrc.getType());
    if (!memType)
      return failure();

    auto loc = op->getLoc();
    auto tileTensorTy = cast<RankedTensorType>(op->getOperand(1).getType());
    auto resultTensorTy = cast<RankedTensorType>(op->getResult(0).getType());

    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    SmallVector<int64_t> tileShapeVec;
    if (tileShapeAttr) {
      tileShapeVec.assign(tileShapeAttr.asArrayRef().begin(),
                          tileShapeAttr.asArrayRef().end());
    } else {
      tileShapeVec.assign(tileTensorTy.getShape().begin(),
                          tileTensorTy.getShape().end());
    }
    ArrayRef<int64_t> tileShape = tileShapeVec;

    auto srcTensorTy = cast<RankedTensorType>(op->getOperand(0).getType());
    auto srcShape = srcTensorTy.getShape();

    if (!triton::gcu::needsSmemRelay(srcTensorTy, tileShape)) {
      auto outputType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(resultTensorTy));
      auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, outputType);
      auto elemOffsets = computeElemOffsets(rewriter, loc, convertedIndex,
                                            srcShape, tileShape);

      auto cpTag = pTagPool.getPrivateSyncTagInfo(op);
      SmallVector<Value> zeroOffsets(
          srcShape.size(), rewriter.create<arith::ConstantIntOp>(loc, 0, 32));
      auto defaultValue = triton::gcu::createConstantZero(
          rewriter, loc, resultTensorTy.getElementType());
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(srcTensorTy);
      rewriter.create<memref_ext::SliceStartOp>(
          loc, output, convertedSrc, zeroOffsets, defaultValue, cpTag.getTag(),
          ValueRange{cpTag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, cpTag.getTag(), ValueRange{cpTag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

      auto tileTotalElems = triton::gcu::getTotalElemsPerThread(tileTensorTy);
      auto dsTag = pTagPool.getPrivateSyncTagInfo(op);
      rewriter.create<memref_ext::DesliceStartOp>(loc, output, convertedTile,
                                                  elemOffsets, dsTag.getTag(),
                                                  ValueRange{dsTag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, dsTag.getTag(), ValueRange{dsTag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, tileTotalElems));

      leaveTritionOp(rewriter, op);
      rewriter.replaceOp(op, output);
      return success();
    }

    // SMEM relay path: check if shared memory is already allocated.
    auto memSpace = memType.getMemorySpace();
    bool isInSharedMem =
        (isa_and_nonnull<mlir::triton::gpu::SharedMemorySpaceAttr>(memSpace)) ||
        (isa_and_nonnull<IntegerAttr>(memSpace) &&
         cast<IntegerAttr>(memSpace).getInt() == 2);

    auto smemBuf = convertedSrc;
    if (!isInSharedMem) {
      // Shared memory not yet allocated (local-mem-optimize didn't fuse).
      // Store data to shared memory first.
      auto lastUser = userAnalysis.getLastUser(op->getResults()[0]);
      auto storeTag = pTagPool.getPrivateSyncTagInfo(op);
      smemBuf = storeToSharedMem(rewriter, storeTag, srcTensorTy, convertedSrc,
                                 /*onlyThread0=*/false, lastUser, userAnalysis,
                                 replaced2Origin);
    }

    auto output =
        emitDesliceToSmem(rewriter, loc, smemBuf, convertedTile, convertedIndex,
                          srcShape, tileShape, tileTensorTy, resultTensorTy,
                          userAnalysis, replaced2Origin, pTagPool, op);

    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, output);
    return success();
  }
};

// ============================================================================
// Lowering: triton_gcu.slice_from_local (SMEM src, always from memdesc)
//
// Created by local-mem-optimize. src is always in shared memory.
// Delegates to emitSliceFromSmem.
// ============================================================================
struct SliceFromLocalOpLowering
    : SharedConversionPattern<triton::gcu::SliceFromLocalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::SliceFromLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());

    auto resultTensorTy = cast<RankedTensorType>(op.getResult().getType());
    auto outputType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultTensorTy));

    auto output = emitSliceFromSmem(
        rewriter, op.getLoc(),
        stripUnrealizedConversionCastOp(adaptor.getSrc()), adaptor.getIndex(),
        op.getSrc().getType().getShape(), op.getTileShape(), resultTensorTy,
        outputType, userAnalysis, replaced2Origin, pTagPool, op);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

// ============================================================================
// Lowering: triton_gcu.deslice_to_local (SMEM src, always from memdesc)
//
// Created by local-mem-optimize. src is always in shared memory.
// Delegates to emitDesliceToSmem.
// ============================================================================
struct DesliceToLocalOpLowering
    : SharedConversionPattern<triton::gcu::DesliceToLocalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::DesliceToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());

    auto tileTensorTy = cast<RankedTensorType>(op.getTile().getType());
    auto resultTensorTy = cast<RankedTensorType>(op.getResult().getType());

    auto output = emitDesliceToSmem(
        rewriter, op.getLoc(),
        stripUnrealizedConversionCastOp(adaptor.getSrc()), adaptor.getTile(),
        adaptor.getIndex(), op.getSrc().getType().getShape(), op.getTileShape(),
        tileTensorTy, resultTensorTy, userAnalysis, replaced2Origin, pTagPool,
        op);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTMemDescIndexOpLowering
    : SharedConversionPattern<triton::gpu::MemDescIndexOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert MemDescIndexOp result to MemRefType");
    auto loc = op.getLoc();
    auto src = stripUnrealizedConversionCastOp(adaptor.getSrc());
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "MemDescIndexOp source is not a MemRefType");
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;

    auto indexValue = adaptor.getIndex();

    auto strideValue = rewriter.create<arith::ConstantIntOp>(
        loc, indexValue.getType(), strides[0]);
    auto finalOffsetValue =
        rewriter.create<arith::MulIOp>(loc, indexValue, strideValue);

    auto elemType = resultType.getElementType();
    auto bpe = elemType.getIntOrFloatBitWidth() / 8;

    // Base pointer of the view's element 0: for a strided view source this
    // folds the view offset into the root allocation pointer (getMemRefBasePtr);
    // for an identity source it is the plain memref2ptr. The per-index
    // byte offset is added below via the ViewOp offset.
    Value ptr = getMemRefBasePtr(rewriter, src);
    MemRefType memType1D =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    auto buffer1D = rewriter.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);

    auto I8Offset = rewriter.create<arith::MulIOp>(
        loc, finalOffsetValue,
        rewriter.create<arith::ConstantIntOp>(loc, indexValue.getType(), bpe));
    auto bufferWithSpace = rewriter.create<memref::MemorySpaceCastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        AffineMap{}, resultType.getMemorySpace()),
        buffer1D);
    auto output = rewriter.create<memref::ViewOp>(
        loc, resultType, bufferWithSpace,
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                            I8Offset),
        ValueRange{});
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

/// Convert an operation's result type through the type converter, failing
/// explicitly when the result is not a MemRefType (instead of
/// dereferencing a null type further down).
static FailureOr<MemRefType> convertResultToMemRef(const TypeConverter &tc,
                                                   Operation *op) {
  auto ty = dyn_cast<MemRefType>(tc.convertType(op->getResult(0).getType()));
  if (!ty)
    return failure();
  return ty;
}

struct TTMemDescSubsliceOpLowering
    : SharedConversionPattern<triton::gpu::MemDescSubsliceOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubsliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType = convertResultToMemRef(*getTypeConverter(), op);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to convert MemDescSubsliceOp result to MemRefType");
    auto loc = op.getLoc();
    auto src = stripUnrealizedConversionCastOp(adaptor.getSrc());

    auto opOffsets = op.getOffsets();
    auto rank = opOffsets.size();
    auto resultShape = resultType->getShape();

    auto sourceType = dyn_cast<MemRefType>(src.getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "MemDescSubsliceOp source is not a MemRefType");
    auto [srcStrides, srcOffset] = sourceType.getStridesAndOffset();
    if (ShapedType::isDynamic(srcOffset))
      return rewriter.notifyMatchFailure(
          op, "MemDescSubsliceOp source has a dynamic offset");
    SmallVector<int64_t> srcStrideVec(srcStrides.begin(), srcStrides.end());
    for (int64_t stride : srcStrideVec)
      if (ShapedType::isDynamic(stride))
        return rewriter.notifyMatchFailure(
            op, "MemDescSubsliceOp source has a dynamic stride");
    // Compute the flat element offset for the StridedLayoutAttr.
    // SubViewOp verifier requires the result type's offset to match
    // the actual offset derived from the offsets operand (relative to the
    // source view's base, so the source view's own offset is included).
    int64_t flatOffset = srcOffset;
    for (unsigned i = 0; i < rank; ++i)
      flatOffset += opOffsets[i] * srcStrides[i];
    auto stridedLayout =
        StridedLayoutAttr::get(rewriter.getContext(), flatOffset, srcStrideVec);
    auto stridedResultTy =
        MemRefType::get(resultShape, resultType->getElementType(),
                        stridedLayout, resultType->getMemorySpace());

    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (unsigned i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.getIndexAttr(opOffsets[i]));
      sizes.push_back(rewriter.getIndexAttr(resultShape[i]));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    auto subview = rewriter.create<memref::SubViewOp>(loc, stridedResultTy, src,
                                                      offsets, sizes, strides);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, subview.getResult());
    return success();
  }
};

struct TTMemDescTransOpLowering
    : SharedConversionPattern<triton::gpu::MemDescTransOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescTransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType = convertResultToMemRef(*getTypeConverter(), op);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to convert MemDescTransOp result to MemRefType");
    auto loc = op.getLoc();
    auto src = stripUnrealizedConversionCastOp(adaptor.getSrc());

    auto resultShape = resultType->getShape();
    auto rank = static_cast<int64_t>(resultType->getRank());
    auto order = op.getOrder();

    // Build a strided result MemRefType with permuted source strides.
    // The type converter's identity AffineMap{} cannot hold non-row-major
    // strides, so we construct StridedLayoutAttr manually with
    // srcStrides[order[i]].
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "MemDescTransOp source is not a MemRefType");
    auto [srcStrides, srcOffset] = sourceType.getStridesAndOffset();
    if (ShapedType::isDynamic(srcOffset))
      return rewriter.notifyMatchFailure(
          op, "MemDescTransOp source has a dynamic offset");
    SmallVector<int64_t> transStrides(rank);
    for (int64_t i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(srcStrides[order[i]]))
        return rewriter.notifyMatchFailure(
            op, "MemDescTransOp source has a dynamic stride");
      transStrides[i] = srcStrides[order[i]];
    }
    // The transposed view addresses the same base buffer at the source
    // view's offset (e.g. trans(subslice) keeps the subslice offset).
    auto stridedLayout =
        StridedLayoutAttr::get(rewriter.getContext(),
                               /*offset=*/srcOffset, transStrides);
    auto stridedResultTy =
        MemRefType::get(resultShape, resultType->getElementType(),
                        stridedLayout, resultType->getMemorySpace());

    // Use ReinterpretCastOp (not SubViewOp) because trans permutes
    // strides, which SubViewOp cannot do (it only slices, preserving
    // source stride order).
    auto offset = rewriter.create<arith::ConstantIndexOp>(loc, srcOffset);
    SmallVector<Value> sizeVals, strideVals;
    for (int64_t i = 0; i < rank; ++i) {
      sizeVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
      strideVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, transStrides[i]));
    }

    auto output = rewriter.create<memref::ReinterpretCastOp>(
        loc, stridedResultTy, src, offset, sizeVals, strideVals);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTMemDescReshapeOpLowering
    : SharedConversionPattern<triton::gpu::MemDescReshapeOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType = convertResultToMemRef(*getTypeConverter(), op);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to convert MemDesc view op result to MemRefType");
    auto loc = op.getLoc();
    auto src = stripUnrealizedConversionCastOp(adaptor.getSrc());

    auto resultShape = resultType->getShape();
    auto rank = static_cast<int64_t>(resultType->getRank());

    auto [resultStrides, resultOffset] = resultType->getStridesAndOffset();
    (void)resultOffset;

    auto sourceType = dyn_cast<MemRefType>(src.getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "MemDescReshapeOp source is not a MemRefType");
    auto [srcStrides, srcOffset] = sourceType.getStridesAndOffset();
    (void)srcStrides;
    if (ShapedType::isDynamic(srcOffset))
      return rewriter.notifyMatchFailure(
          op, "MemDescReshapeOp source has a dynamic offset");

    // Carry a non-zero source view offset in a strided result layout: the
    // identity AffineMap{} cannot express it. The common path (reshape of
    // the root 1D allocation) has offset 0 and keeps the identity type.
    MemRefType outputType = *resultType;
    if (srcOffset != 0)
      outputType =
          MemRefType::get(resultShape, resultType->getElementType(),
                          StridedLayoutAttr::get(rewriter.getContext(),
                                                 srcOffset, resultStrides),
                          resultType->getMemorySpace());

    auto offset = rewriter.create<arith::ConstantIndexOp>(loc, srcOffset);
    SmallVector<Value> sizeVals, strideVals;
    for (int64_t i = 0; i < rank; i++) {
      sizeVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
      strideVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, resultStrides[i]));
    }

    auto output = rewriter.create<memref::ReinterpretCastOp>(
        loc, outputType, src, offset, sizeVals, strideVals);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTMemDescReinterpretOpLowering
    : SharedConversionPattern<triton::gpu::MemDescReinterpretOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescReinterpretOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType = convertResultToMemRef(*getTypeConverter(), op);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to convert MemDesc view op result to MemRefType");
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();

    auto resultShape = resultType->getShape();
    auto rank = static_cast<int64_t>(resultType->getRank());

    auto [resultStrides, resultOffset] = resultType->getStridesAndOffset();
    (void)resultOffset;

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> sizeVals, strideVals;
    for (int64_t i = 0; i < rank; i++) {
      sizeVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
      strideVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, resultStrides[i]));
    }

    auto output = rewriter.create<memref::ReinterpretCastOp>(
        loc, *resultType, src, zero, sizeVals, strideVals);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void mlir::triton::populateTTSmemOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::DTETagPool &pTagPool) {
  patterns.add<TTSmemLocalAllocOpLowering,
               TTSmemCopyGlobalToLocalOpLowering,
               TTSmemGatherGlobalToLocalOpLowering,
               TTLocalLoadOpLowering,
               TTLocalStoreOpLowering,
               TTMemDescIndexOpLowering,
               TTMemDescSubsliceOpLowering,
               TTMemDescTransOpLowering,
               TTMemDescReshapeOpLowering,
               TTMemDescReinterpretOpLowering,
               MemDescToPtrOpLowering,
               SliceFromLocalOpLowering,
               DesliceToLocalOpLowering>(converter, patterns.getContext(),
                                          userAnalysis, replaced2Origin,
                                          pTagPool);
  patterns.add<TleExtractTileLowering,
               TleInsertTileLowering>(converter, patterns.getContext(),
                                       userAnalysis, replaced2Origin, pTagPool);
}
