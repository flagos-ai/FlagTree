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

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

    auto output = rewriter.create<memref::AllocOp>(alloc.getLoc(), memrefType);

    leaveTritionOp(rewriter, alloc.getOperation());
    rewriter.replaceOp(alloc, output);
    return success();
  }
};

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

    auto output = adaptor.getDstMem();
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!outputType)
      return failure();

    auto elemType = outputType.getElementType();
    int64_t rank = outputType.getRank();

    auto tagShare = getShareDTETag(rewriter, loadOp);
    triton::gcu::TagInfo tag(tagShare, zero, true);

    auto defaultValue =
        loadOp.getDefaultValue()
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
    auto total_size =
        rewriter
            .create<scf::IfOp>(
                loc, shapeCheck,
                [&](OpBuilder builder, Location loc) {
                  Value load_size =
                      ConfigGcuLoad(builder, loc, pTagPool, output, loadOp,
                                    outputType, adaptor.getPtr(),
                                    adaptor.getStrides(), adaptor.getShape(),
                                    defaultValue, tag, /*IsShareOutput=*/true);
                  builder.create<scf::YieldOp>(loc, ValueRange{load_size});
                },
                [&](OpBuilder &builder, Location loc) {
                  builder.create<scf::YieldOp>(loc, ValueRange{zero});
                })
            .getResult(0);

    // IsShareOutput wait: only thread0 waits, then barrier
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto isAll = rewriter.create<arith::AndIOp>(loc, isThread0, shapeCheck);
    rewriter.create<scf::IfOp>(
        loc, isAll, [&](OpBuilder builder, Location loc) {
          WaitGcuLoadStore(builder, loc, tag, total_size);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);

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
    unsigned K; // inner dim per warp
    unsigned M; // outer dim per warp (1 for flat 1D)
    unsigned totalNumElems;
    unsigned totalCols; // full smem row width
    bool isColSplit;

    Value flatDst;
    Value warpFlatBase;
    Value masksPtr;         // nullptr when no mask
    Value vtOther;          // passthru vector
    VectorType vtType;      // <VL x elemType>
    VectorType maskVecType; // <VL x i1>
    Value totalColsVal;
    Value innerDimVal; // ConstantIndexOp(K)

    // For vector::GatherOp tail path
    Value srcFlatI64;      // src ptrs as flat i64 memref
    Value baseI64;         // first ptr value (i64 scalar)
    Value baseMemref;      // base dynamic memref for GatherOp
    Value baseI64Vec;      // broadcast of baseI64 to <VL x i64>
    Value bpeVec;          // broadcast of bpe to <VL x i64>
    VectorType i64VecType; // <VL x i64>
    VectorType i32VecType; // <VL x i32>
    Value zeroI64Passthru; // <VL x 0_i64>
  };

  LogicalResult buildContext(ConversionPatternRewriter &rewriter, Location loc,
                             triton::gcu::GatherGlobalToLocalOp gatherOp,
                             OpAdaptor adaptor, G2LEmitContext &ctx) const {
    auto dstMemref = dyn_cast<MemRefType>(adaptor.getDstMem().getType());
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
    ctx.flatDst = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, adaptor.getDstMem(), static_cast<int64_t>(0),
        ArrayRef<int64_t>{flatTotal}, ArrayRef<int64_t>{1LL});

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
      unsigned rowDelta;  // row offset from loop base row
      unsigned colOffset; // column offset within the row
      bool isTail;        // whether this slot is a tail chunk
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
      tag = pTagPool.trygGetAsyncTagInfo(op);
    } else {
      tag = pTagPool.getSyncTagInfo(op);
    }
    if (tag.isAsync()) {
      pTagPool.setMap(firstUser.first, tag);
    }
    // share to Distributed
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // copy to local
      ModuleOp builtinModule = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gcu::getNumWarps(builtinModule);
      if (numWarps > 1) {
        auto output = loadFromSharedMem(
            rewriter, tag, op.getResult().getType(), adaptor.getSrc(), false,
            lastUser, firstUser, userAnalysis, replaced2Origin);
        leaveTritionOp(rewriter, op.getOperation());
        rewriter.replaceOp(op, output);
        return success();
      } else {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    } else if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      auto loc = op.getLoc();
      auto srcMemRef = dyn_cast<MemRefType>(adaptor.getSrc().getType());
      if (!srcMemRef)
        return failure();

      auto numElems = triton::gcu::getElemsPerThread(op.getType());
      auto warpIds = getWarpIds(rewriter, loc, op.getType());
      unsigned rank = srcMemRef.getRank();

      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (unsigned i = 0; i < rank; ++i) {
        Value offset = rewriter.create<arith::MulIOp>(
            loc, rewriter.create<arith::ConstantIndexOp>(loc, numElems[i]),
            warpIds[i]);
        offsets.push_back(offset);
        sizes.push_back(rewriter.getIndexAttr(numElems[i]));
        strides.push_back(rewriter.getIndexAttr(1));
      }

      auto resultType =
          cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
      auto subview = rewriter.create<memref::SubViewOp>(
          loc, resultType, adaptor.getSrc(), offsets, sizes, strides);

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

#ifdef ENABLE_TLE
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
    auto loc = op.getLoc();
    auto srcMemRef = dyn_cast<MemRefType>(adaptor.getSrc().getType());
    if (!srcMemRef)
      return failure();

    auto elemTy = srcMemRef.getElementType();
    auto gcuPtrTy = mlir::gcu::PtrType::get(op.getContext(), elemTy);
    auto ptr = rewriter.create<mlir::gcu::MemRefToPtrOp>(loc, gcuPtrTy,
                                                         adaptor.getSrc());

    rewriter.replaceOp(op, ptr.getResult());
    return success();
  }
};
#endif

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
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;

    auto indexValue = adaptor.getIndex();

    auto strideValue = rewriter.create<arith::ConstantIntOp>(
        loc, indexValue.getType(), strides[0]);
    auto finalOffsetValue =
        rewriter.create<arith::MulIOp>(loc, indexValue, strideValue);

    auto elemType = resultType.getElementType();
    auto bpe = elemType.getIntOrFloatBitWidth() / 8;
    auto elementType = resultType.getElementType();

    int64_t size = 1;
    for (int i = 0; i < sourceType.getRank(); i++) {
      size *= sourceType.getShape()[i];
    }

    MemRefType flatType = MemRefType::get({size}, elementType, AffineMap{},
                                          resultType.getMemorySpace());
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value flatBuffer = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, src, zero,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, size)},
        ValueRange{one});
    auto ptrType = gcu::PtrType::get(getContext(), elementType);
    Value ptr = rewriter.create<gcu::MemRefToPtrOp>(loc, ptrType, flatBuffer);
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

} // namespace

void mlir::triton::populateTTSmemOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TTSmemLocalAllocOpLowering, TTSmemCopyGlobalToLocalOpLowering,
               TTSmemGatherGlobalToLocalOpLowering, TTLocalLoadOpLowering,
               TTMemDescIndexOpLowering>(converter, patterns.getContext(),
                                         userAnalysis, replaced2Origin,
                                         pTagPool);
#ifdef ENABLE_TLE
  patterns.add<MemDescToPtrOpLowering>(converter, patterns.getContext(),
                                       userAnalysis, replaced2Origin, pTagPool);
#endif
}
