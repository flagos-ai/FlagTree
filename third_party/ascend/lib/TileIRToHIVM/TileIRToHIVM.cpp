/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
// TileIRToHIVM — Lowers TileIR dialect ops to HIVM (Ascend NPU) dialect ops.
//
// Strategy: greedy rewrite runs patterns iteratively to fixed point.
//   Step 1: tile.alloc → memref.alloc (produces memref with #hivm.address_space)
//   Step 2: tile.to_tensor → RAUW: replace all uses of result with src operand,
//           then erase. The src is now memref (from Step 1), consumers get
//           memref instead of tensor, bridged by UnrealizedConversionCast.
//   Step 3: tile.copy → hivm.copy (memref in, memref out)
//   Step 4: tile.load/store → hivm.load/store
//   Step 5: sync ops → hivm sync ops
//===----------------------------------------------------------------------===//

#include "ascend/include/TileIRToHIVM/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
namespace tile = mlir::triton::tile;
namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TILEIRTOHIVM
#include "ascend/include/TileIRToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

// =============================================================================
// Helpers
// =============================================================================
static hivm::AddressSpace mapMemSpaceToHIVM(tile::MemorySpace tileSpace) {
  switch (tileSpace) {
  case tile::MemorySpace::GM:  return hivm::AddressSpace::GM;
  case tile::MemorySpace::L1:  return hivm::AddressSpace::L1;
  case tile::MemorySpace::L0A: return hivm::AddressSpace::L0A;
  case tile::MemorySpace::L0B: return hivm::AddressSpace::L0B;
  case tile::MemorySpace::L0C: return hivm::AddressSpace::L0C;
  case tile::MemorySpace::UB:  return hivm::AddressSpace::UB;
  }
  llvm_unreachable("unknown TileIR memory space");
}

static MemRefType convertBufToMemRef(tile::BufType bufTy) {
  auto shape = bufTy.getShape();
  auto elemTy = bufTy.getElementType();
  auto space = bufTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(llvm::SmallVector<int64_t>(shape), elemTy,
                         MemRefLayoutAttrInterface{},
                         hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

static MemRefType convertTensorToMemRef(tile::TensorType tensorTy) {
  auto shape = tensorTy.getShape();
  auto elemTy = tensorTy.getElementType();
  auto space = tensorTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(llvm::SmallVector<int64_t>(shape), elemTy,
                         MemRefLayoutAttrInterface{},
                         hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

/// Get the underlying memref value, converting tile/builtin tensor types to memref.
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
  else if (auto rankedTy = dyn_cast<RankedTensorType>(val.getType()))
    targetTy = MemRefType::get(rankedTy.getShape(), rankedTy.getElementType());
  else if (auto ptrTy = dyn_cast<triton::PointerType>(val.getType())) {
    // tt.ptr<tensor<...>> → memref<...>
    auto pointeeTy = ptrTy.getPointeeType();
    if (auto rankedTy = dyn_cast<RankedTensorType>(pointeeTy))
      targetTy = MemRefType::get(rankedTy.getShape(), rankedTy.getElementType());
    else
      targetTy = MemRefType::get({ShapedType::kDynamic}, pointeeTy);
  } else
    return val;
  return rewriter.create<UnrealizedConversionCastOp>(
      val.getLoc(), targetTy, val)->getResult(0);
}

static hivm::PIPE mapPipe(int64_t tilePipe) {
  switch (static_cast<tile::Pipe>(tilePipe)) {
  case tile::Pipe::PIPE_M:    return hivm::PIPE::PIPE_M;
  case tile::Pipe::PIPE_V:    return hivm::PIPE::PIPE_V;
  case tile::Pipe::PIPE_MTE1: return hivm::PIPE::PIPE_MTE1;
  case tile::Pipe::PIPE_MTE2: return hivm::PIPE::PIPE_MTE2;
  case tile::Pipe::PIPE_MTE3: return hivm::PIPE::PIPE_MTE3;
  case tile::Pipe::PIPE_FIX:  return hivm::PIPE::PIPE_FIX;
  case tile::Pipe::PIPE_S:    return hivm::PIPE::PIPE_S;
  }
  llvm_unreachable("unknown TileIR pipe");
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
// Step 2: tile.to_tensor → UnrealizedConversionCast (memref → tensor)
// =============================================================================
struct TileToTensorEliminate : OpRewritePattern<tile::ToTensorOp> {
  using OpRewritePattern<tile::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::ToTensorOp op,
                                PatternRewriter &rewriter) const final {
    // The src is now memref (from Step 1).  The result type is tensor<>.
    // Bridge the gap with UnrealizedConversionCast: memref → tensor.
    Value src = op.getSrc();
    auto resultTy = op.getResult().getType();
    if (src.getType() != resultTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), resultTy, src);
      rewriter.replaceOp(op, cast->getResult(0));
    } else {
      rewriter.replaceOp(op, src);
    }
    return success();
  }
};

// =============================================================================
// Step 3: tile.copy → hivm.copy
//   Skip tile.copy with !tt.ptr source — those are GM→local DMAs that must be
//   handled after !tt.ptr→memref conversion (triton_to_linalg_incubated).
// =============================================================================
struct TileCopyToHIVM : OpRewritePattern<tile::CopyOp> {
  using OpRewritePattern<tile::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::CopyOp op,
                                PatternRewriter &rewriter) const final {
    // Skip !tt.ptr sources — they need to be lowered after tt.ptr→memref.
    if (isa<triton::PointerType>(op.getSrc().getType()))
      return failure();
    Value srcMem = getAsMemRef(op.getSrc(), rewriter);
    Value dstMem = getAsMemRef(op.getDst(), rewriter);
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
  LogicalResult matchAndRewrite(tile::SetFlagOp op, PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SetFlagOp>(op,
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx, mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TileWaitFlagToHIVM : OpRewritePattern<tile::WaitFlagOp> {
  using OpRewritePattern<tile::WaitFlagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::WaitFlagOp op, PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::WaitFlagOp>(op,
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx, mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TilePipeBarrierToHIVM : OpRewritePattern<tile::PipeBarrierOp> {
  using OpRewritePattern<tile::PipeBarrierOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::PipeBarrierOp op, PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::PipeBarrierOp>(
        op, hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getPipe()))));
    return success();
  }
};

struct TileCubeWaitToHIVM : OpRewritePattern<tile::CubeWaitOp> {
  using OpRewritePattern<tile::CubeWaitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::CubeWaitOp op, PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SyncBlockWaitOp>(op,
        hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
        OpFoldResult(rewriter.getIndexAttr(0)));
    return success();
  }
};

struct TileGmOffsetToHIVM : OpRewritePattern<tile::GmOffsetOp> {
  using OpRewritePattern<tile::GmOffsetOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::GmOffsetOp op, PatternRewriter &rewriter) const final {
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

// =============================================================================
// Pass
// =============================================================================
namespace {
struct TileIRToHIVMPass : public mlir::triton::impl::TileIRToHIVMBase<TileIRToHIVMPass> {
  void runOnOperation() override;
};
} // namespace

void TileIRToHIVMPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy rewrite to iteratively apply patterns to fixed point.
  // This avoids the type-conversion complexity of the dialect conversion
  // framework: tile.alloc → memref.alloc replaces !tile.buf with memref,
  // then tile.copy (now seeing memref operands) → hivm.copy, etc.
  RewritePatternSet patterns(&getContext());
  patterns.add<TileAllocToMemRef, TileToTensorEliminate, TileCopyToHIVM,
               TileLoadToHIVM, TileStoreToHIVM,
               TileSetFlagToHIVM, TileWaitFlagToHIVM, TilePipeBarrierToHIVM,
               TileCubeWaitToHIVM, TileGmOffsetToHIVM>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTileIRToHIVMPass() {
  return std::make_unique<TileIRToHIVMPass>();
}