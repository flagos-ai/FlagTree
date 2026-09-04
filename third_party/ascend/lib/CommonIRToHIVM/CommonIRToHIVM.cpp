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
//   Step 1: tile.alloc → memref.alloc (produces memref with
//   #hivm.address_space) Step 2: tile.to_tensor → RAUW: replace all uses of
//   result with src operand,
//           then erase. The src is now memref (from Step 1), consumers get
//           memref instead of tensor, bridged by UnrealizedConversionCast.
//   Step 3: tile.copy → hivm.copy (tile.* src, on-chip DMA)
//                    or memref.copy (!tt.ptr src, GM↔local DMA; the ptr is
//                    bridged to memref via UnrealizedConversionCast)
//           tile.store_tensor → hivm.copy.
//   Step 4: tile.load/store → hivm.load/store
//   Step 5: sync ops → hivm sync ops
//===----------------------------------------------------------------------===//

#include "ascend/include/CommonIRToHIVM/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
namespace tile = mlir::triton::tile;
namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
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
    auto resultTy = op.getResult().getType();
    if (auto t = dyn_cast<tile::TensorType>(resultTy)) {
      auto memrefTy = convertTensorToMemRef(t);
      rewriter.replaceOpWithNewOp<hivm::LoadOp>(op, memrefTy, op.getOperand());
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
    Value src = getAsMemRef(op.getOperand(0), rewriter);
    rewriter.replaceOpWithNewOp<hivm::StoreOp>(op, Type(), src,
                                               op.getOperand(1));
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
