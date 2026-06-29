/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

//===----------------------------------------------------------------------===//
// EraseLinalgCasts -- Lower the unrealized_conversion_cast ops left around by
// the TileIRToHIVM pass into legitimate MLIR bridges, so the subsequent
// triton-to-linalg-incubated conversion produces cast-free linalg-dialect IR.
//
// TileIRToHIVM leaves two flavours of unrealized casts in the module:
//
//   Pattern A: !tt.ptr<tensor<TxS,E>> -> memref<TxS,E>
//     used to feed memref.copy %cast, %dst (GM -> on-chip DMA stand-in).
//     Replaced with:
//        %t = tt.load %ptr : !tt.ptr<tensor<TxS,E>>
//        %m = bufferization.to_memref %t : memref<TxS,E>
//        memref.copy %m, %dst
//     `triton-to-linalg-incubated` knows how to lower tt.load + a derived
//     make_tensor_ptr into a memref.subview/copy chain, after which the
//     temporary bufferization.to_memref folds away.
//
//   Pattern B: memref<TxS,E, #space> -> tensor<TxS,E>
//     used to feed tt.dot from on-chip allocations.  Replaced with:
//        %g = memref.memory_space_cast %m : memref<TxS,E,#space> to memref<TxS,E>
//        %t = bufferization.to_tensor %g restrict
//     (when the source memref already has the default address space, the
//      memory_space_cast is skipped).
//
// After running these two patterns we additionally invoke
// reconcileUnrealizedCasts on the module to remove any remaining one-to-one
// cast chains where the source and destination types coincide -- those appear
// after the linalg pass runs.  This combination makes the post-linalg module
// strictly free of `builtin.unrealized_conversion_cast`.
//===----------------------------------------------------------------------===//

#include "ascend/include/EraseLinalgCasts/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ERASELINALGCASTS
#include "ascend/include/EraseLinalgCasts/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static bool isOneToOne(UnrealizedConversionCastOp op) {
  return op.getInputs().size() == 1 && op->getNumResults() == 1;
}

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
    Value asMemref =
        rewriter.create<bufferization::ToMemrefOp>(loc, outMemref, loaded);
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
    Value src = in;
    // Drop any non-default address space attribute (linalg / bufferization
    // require the default address space on the memref feeding to_tensor).
    if (memrefTy.getMemorySpace()) {
      auto generic = MemRefType::get(memrefTy.getShape(),
                                     memrefTy.getElementType(),
                                     memrefTy.getLayout());
      src = rewriter.create<memref::MemorySpaceCastOp>(loc, generic, src);
    }
    // %t = bufferization.to_tensor %g restrict
    Value asTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorOut, src, /*restrict=*/true, /*writable=*/false);
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
struct FoldRoundtripCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
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
// Pass
//===----------------------------------------------------------------------===//
struct EraseLinalgCastsPass
    : public triton::impl::EraseLinalgCastsBase<EraseLinalgCastsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<LowerPtrTensorCastToMemref, LowerMemrefCastToTensor,
                   FoldRoundtripCast>(&getContext());
      if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    // Finally, fold any residual A<->B unrealized cast chains.
    SmallVector<UnrealizedConversionCastOp> casts;
    module->walk(
        [&](UnrealizedConversionCastOp c) { casts.push_back(c); });
    reconcileUnrealizedCasts(casts);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createEraseLinalgCastsPass() {
  return std::make_unique<EraseLinalgCastsPass>();
}
