//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// Throughout the conversion process, we convert !tt.ptr -> {!ptr.ptr or
// memref<*>}. This process leaves around unrealized_conversion_cast ops between
// these types. We want to remove these unrealized casts and use the proper
// conversion ops in the PtrDialect: to_memref or from_memref. To do this, we
// use a pattern that simplifies the chain of conversions by removing
// intermediate conversion cast ops. At the end, we are left with just pointer
// to memref or vice versa. We then convert the unrealized cast to to_memref or
// from_memref accordingly.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Utils/MemorySpaceUtils.h"

#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace triton;

static Attribute getMemorySpaceForMemref(BaseMemRefType memrefType) {
  if (auto space = memrefType.getMemorySpace())
    return space;
  // Fallback: when the memref carries no memory space (nullptr),
  // use the default Global(0) space.
  return mlir::triton::getDefaultBridgeMemorySpace(memrefType.getContext());
}

static ptr::PtrType getPtrTypeForMemref(BaseMemRefType memrefType) {
  auto space = getMemorySpaceForMemref(memrefType);
  auto spaceIface = dyn_cast<ptr::MemorySpaceAttrInterface>(space);
  assert(spaceIface && "memory space must implement MemorySpaceAttrInterface");
  return ptr::PtrType::get(memrefType.getContext(), spaceIface);
}

static MemRefType getRankedMemrefTypeForPtrCast(BaseMemRefType memrefType) {
  // BUGFIX: Use dynamic size instead of hardcoded size=1 to fix masked_select
  // bug where tensor<256xi8> was incorrectly treated as having only 1 element.
  return MemRefType::get({ShapedType::kDynamic}, memrefType.getElementType(),
                         AffineMap(), getMemorySpaceForMemref(memrefType));
}

static MemRefType cloneMemRefWithMemorySpace(MemRefType memrefType,
                                             Attribute memorySpace) {
  return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                         memrefType.getLayout(), memorySpace);
}

static Attribute getMemorySpaceForPtr(Type ptrType) {
  if (auto ptrPtrType = dyn_cast<ptr::PtrType>(ptrType))
    return ptrPtrType.getMemorySpace();
  // Fallback: non-ptr::PtrType (e.g. triton::PointerType) defaults to
  // the default Global(0) space, consistent with getDefaultBridgeMemorySpace().
  return mlir::triton::getDefaultBridgeMemorySpace(ptrType.getContext());
}

static Type getMemrefElementTypeForPtrCast(Type ptrType,
                                           Type fallbackElemType) {
  return fallbackElemType;
}

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_RECONCILEPTRCASTS
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}

struct MemrefCastConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  MemrefCastConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op))
      return failure();

    Value input = op.getInputs().front();
    Type inputTy = input.getType();
    Type resultTy = op.getResult(0).getType();

    auto inputMemrefTy = dyn_cast<BaseMemRefType>(inputTy);
    auto resultMemrefTy = dyn_cast<BaseMemRefType>(resultTy);
    if (!inputMemrefTy || !resultMemrefTy)
      return failure();

    if (inputTy == resultTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto rankedInputTy = dyn_cast<MemRefType>(inputTy)) {
      if (auto rankedResultTy = dyn_cast<MemRefType>(resultTy)) {
        if (memref::CastOp::areCastCompatible(rankedInputTy, rankedResultTy)) {
          rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultTy,
                                                      input);
          return success();
        }
        // When the only incompatibility is memory space (e.g. different
        // xsmt::MemorySpaceAttr scopes from different conversion paths),
        // bridge through ptr::ToPtrOp + ptr::FromPtrOp to reinterpret the
        // pointer while preserving the data address.
        //
        // ptr.from_ptr requires the ptr memory space and the output memref
        // memory space to be the *same* Attribute.  We therefore use the
        // **input** memory space throughout the bridge chain (ToPtrOp →
        // FromPtrOp → ReinterpretCastOp) and produce a memref whose memory
        // space matches the input.  Downstream passes (LLVM lowering) treat
        // all address spaces uniformly, so the concrete IntegerAttr value
        // is irrelevant at that stage.
        if (rankedInputTy.getElementType() == rankedResultTy.getElementType()) {
          auto loc = op.getLoc();
          auto inputSpace = getMemorySpaceForMemref(rankedInputTy);
          auto inputSpaceIface =
              dyn_cast<ptr::MemorySpaceAttrInterface>(inputSpace);
          assert(inputSpaceIface &&
                 "memory space must implement MemorySpaceAttrInterface");
          auto inputPtrTy =
              ptr::PtrType::get(loc.getContext(), inputSpaceIface);
          auto toPtr = ptr::ToPtrOp::create(rewriter, loc, inputPtrTy, input);
          // Use inputSpace for FromPtrOp so that ptr and memref memory
          // spaces match (required by ptr.from_ptr verification).
          auto bridgeMemrefTy = MemRefType::get(
              {1}, rankedResultTy.getElementType(), AffineMap(), inputSpace);
          auto fromPtr = ptr::FromPtrOp::create(rewriter, loc, bridgeMemrefTy,
                                                toPtr, Value());
          // Reinterpret to match the target shape/strides/layout, still
          // keeping the input memory space.
          SmallVector<OpFoldResult> sizes;
          SmallVector<OpFoldResult> strides;
          for (int64_t i = 0, e = rankedResultTy.getRank(); i < e; ++i) {
            sizes.push_back(
                ShapedType::isDynamic(rankedResultTy.getDimSize(i))
                    ? rewriter.getIndexAttr(1)
                    : rewriter.getIndexAttr(rankedResultTy.getDimSize(i)));
            strides.push_back(rewriter.getIndexAttr(1));
          }
          // Build the final type with the input memory space and the
          // target shape so that all downstream consumers see a consistent
          // memory space.
          auto identityResultTy = MemRefType::get(
              rankedResultTy.getShape(), rankedResultTy.getElementType(),
              AffineMap(), inputSpace);
          auto reinterpreted = memref::ReinterpretCastOp::create(
              rewriter, loc, identityResultTy, fromPtr,
              rewriter.getIndexAttr(0), sizes, strides);
          // Replace all uses directly.  When inputSpace differs from the
          // TypeConverter-requested space (e.g. IntegerAttr(3) vs
          // memory space scopes), the difference is benign for LLVM lowering
          // and avoids an illegal memref.cast across memory spaces.
          rewriter.replaceAllUsesWith(op.getResult(0), reinterpreted);
          rewriter.eraseOp(op);
          return success();
        }
        return failure();
      }
      if (auto unrankedResultTy = dyn_cast<UnrankedMemRefType>(resultTy)) {
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedResultTy,
                                                    input);
        return success();
      }
      return failure();
    }

    if (auto unrankedInputTy = dyn_cast<UnrankedMemRefType>(inputTy)) {
      if (auto rankedResultTy = dyn_cast<MemRefType>(resultTy)) {
        if (unrankedInputTy.getElementType() !=
            rankedResultTy.getElementType()) {
          auto loc = op.getLoc();
          auto rankedInputType = getRankedMemrefTypeForPtrCast(unrankedInputTy);
          // BUGFIX: Use large size instead of 1 to allow access to all elements
          auto rankedInput = memref::ReinterpretCastOp::create(
              rewriter, loc, rankedInputType, input, rewriter.getIndexAttr(0),
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0x7FFFFFFF)},
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
          auto ptrType = getPtrTypeForMemref(rankedInputType);
          auto toPtr =
              ptr::ToPtrOp::create(rewriter, loc, ptrType, rankedInput);

          auto resultMemorySpace = getMemorySpaceForMemref(rankedResultTy);
          auto oneDimResultType = MemRefType::get(
              {ShapedType::kDynamic}, rankedResultTy.getElementType(),
              AffineMap(), resultMemorySpace);
          auto fromPtr = ptr::FromPtrOp::create(rewriter, loc, oneDimResultType,
                                                toPtr, Value());

          SmallVector<OpFoldResult> sizes;
          SmallVector<OpFoldResult> strides;
          for (int64_t i = 0, e = rankedResultTy.getRank(); i < e; ++i) {
            sizes.push_back(
                ShapedType::isDynamic(rankedResultTy.getDimSize(i))
                    ? rewriter.getIndexAttr(1)
                    : rewriter.getIndexAttr(rankedResultTy.getDimSize(i)));
            strides.push_back(rewriter.getIndexAttr(1));
          }

          auto identityResultTy = MemRefType::get(
              rankedResultTy.getShape(), rankedResultTy.getElementType(),
              AffineMap(), rankedResultTy.getMemorySpace());
          auto reinterpreted = memref::ReinterpretCastOp::create(
              rewriter, loc, identityResultTy, fromPtr,
              rewriter.getIndexAttr(0), sizes, strides);
          if (reinterpreted.getType() == rankedResultTy) {
            rewriter.replaceOp(op, reinterpreted.getResult());
          } else {
            if (!memref::CastOp::areCastCompatible(reinterpreted.getType(),
                                                   rankedResultTy))
              return failure();
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultTy,
                                                        reinterpreted);
          }
          return success();
        }
        if (!memref::CastOp::areCastCompatible(unrankedInputTy, rankedResultTy))
          return failure();
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultTy, input);
        return success();
      }
      if (auto unrankedResultTy = dyn_cast<UnrankedMemRefType>(resultTy)) {
        // If element types differ (e.g. from tt.bitcast ptr<i1> -> ptr<i8>),
        // bridge through ptr dialect since memref.cast cannot change element
        // type.
        if (unrankedInputTy.getElementType() !=
            unrankedResultTy.getElementType()) {
          auto loc = op.getLoc();
          // Cast unranked input to ranked memref with dynamic size
          auto rankedInputType = getRankedMemrefTypeForPtrCast(unrankedInputTy);
          // BUGFIX: Use large size instead of 1
          auto rankedInput = memref::ReinterpretCastOp::create(
              rewriter, loc, rankedInputType, input, rewriter.getIndexAttr(0),
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0x7FFFFFFF)},
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
          // Convert to opaque ptr
          auto ptrType = getPtrTypeForMemref(rankedInputType);
          auto toPtr =
              ptr::ToPtrOp::create(rewriter, loc, ptrType, rankedInput);
          // Convert back to ranked memref with new element type
          auto rankedResultType =
              getRankedMemrefTypeForPtrCast(unrankedResultTy);
          auto fromPtr = ptr::FromPtrOp::create(rewriter, loc, rankedResultType,
                                                toPtr, Value());
          // Cast ranked back to unranked
          rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedResultTy,
                                                      fromPtr);
          return success();
        }
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedResultTy,
                                                    input);
        return success();
      }
    }

    return failure();
  }
};

struct SimplifyUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  SimplifyUnrealizedCast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto in = op.getInputs().front();

    if (auto unrealizedCast = in.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!isOneToOneCast(unrealizedCast)) {
        return failure();
      }

      auto prevInput = unrealizedCast.getInputs().front();
      auto newCast = UnrealizedConversionCastOp::create(
          rewriter, op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

      rewriter.replaceOp(op, newCast);
      return success();
    }
    return failure();
  }
};

struct FromMemrefConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  FromMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    auto input = op.getInputs().front();
    auto unrankedInput = dyn_cast<UnrankedMemRefType>(input.getType());
    auto output = op.getResult(0);
    auto outType = output.getType();

    if (unrankedInput && isa<triton::PointerType, ptr::PtrType>(outType)) {
      // Derive the target ptr type directly from the cast output.
      // We cannot rely on the memref memory space because earlier passes
      // may have left it empty.  The output ptr type already carries the
      // correct memory space.
      ptr::PtrType targetPtrType;
      if (auto ptrOut = dyn_cast<ptr::PtrType>(outType)) {
        targetPtrType = ptrOut;
      } else {
        // triton::PointerType — fall back to default bridge space.
        auto space =
            mlir::triton::getDefaultBridgeMemorySpace(rewriter.getContext());
        auto spaceIface = dyn_cast<ptr::MemorySpaceAttrInterface>(space);
        assert(spaceIface &&
               "memory space must implement MemorySpaceAttrInterface");
        targetPtrType = ptr::PtrType::get(rewriter.getContext(), spaceIface);
      }

      // Build a ranked memref with dynamic size whose memory space matches
      // the target ptr.  Use the ptr's memory space attribute directly so
      // that ptr.to_ptr verification sees identical spaces on both sides.
      // BUGFIX: Use dynamic size instead of hardcoded size=1 to fix
      // masked_select bug where tensor<256xi8> was incorrectly treated as
      // having only 1 element.
      Attribute ptrMemSpace = targetPtrType.getMemorySpace();
      auto rankedType = MemRefType::get({ShapedType::kDynamic},
                                        unrankedInput.getElementType(),
                                        AffineMap(), ptrMemSpace);
      // CRITICAL FIX: Use a very large size (INT32_MAX) instead of 1, so the
      // memref can be accessed at any valid index. The actual bounds checking
      // happens in the linalg.generic based on n_elements parameter.
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(0x7FFFFFFF)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
      auto rankedMemref = memref::ReinterpretCastOp::create(
          rewriter, op.getLoc(), rankedType, input, rewriter.getIndexAttr(0),
          sizes, strides);
      auto memrefToPtr = ptr::ToPtrOp::create(rewriter, op->getLoc(),
                                              targetPtrType, rankedMemref);

      rewriter.replaceAllUsesWith(output, memrefToPtr);
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
  }
};

struct ToMemrefConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  ToMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto input = op.getInputs().front();
    auto inType = input.getType();
    auto output = op.getResult(0);
    auto outRankedMemrefType = dyn_cast<MemRefType>(output.getType());
    auto outUnrankedMemrefType = dyn_cast<UnrankedMemRefType>(output.getType());
    if (isa<triton::PointerType, ptr::PtrType>(inType) && outRankedMemrefType) {
      // Derive the memory space from the input ptr type so that
      // ptr.from_ptr verification sees matching spaces on both sides.
      Attribute outMemSpace = getMemorySpaceForPtr(inType);

      auto elemType = getMemrefElementTypeForPtrCast(
          inType, outRankedMemrefType.getElementType());
      // BUGFIX: If target type has dynamic dimensions, create FromPtrOp with
      // dynamic size to avoid hardcoding size=1 (masked_select bug fix)
      bool hasDynamicDims =
          llvm::any_of(outRankedMemrefType.getShape(),
                       [](int64_t dim) { return ShapedType::isDynamic(dim); });

      auto ptrToMemrefType =
          hasDynamicDims
              ? MemRefType::get({ShapedType::kDynamic}, elemType, AffineMap(),
                                outMemSpace)
              : MemRefType::get({1}, elemType, AffineMap(), outMemSpace);

      auto ptrToMemref = ptr::FromPtrOp::create(
          rewriter, op->getLoc(), ptrToMemrefType, input, Value());

      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> newStrides;
      for (int64_t i = 0, e = outRankedMemrefType.getRank(); i < e; ++i) {
        sizes.push_back(
            ShapedType::isDynamic(outRankedMemrefType.getDimSize(i))
                ? rewriter.getIndexAttr(
                      0x7FFFFFFF) // BUGFIX: Use large size instead of 1
                : rewriter.getIndexAttr(outRankedMemrefType.getDimSize(i)));
        newStrides.push_back(rewriter.getIndexAttr(1));
      }
      auto identityRankedMemrefType = MemRefType::get(
          outRankedMemrefType.getShape(), elemType, AffineMap(), outMemSpace);
      auto newRankedMemref = memref::ReinterpretCastOp::create(
          rewriter, op->getLoc(), identityRankedMemrefType, ptrToMemref,
          rewriter.getIndexAttr(0), sizes, newStrides);

      if (output.getType() == newRankedMemref.getType()) {
        rewriter.replaceAllUsesWith(output, newRankedMemref);
      } else {
        auto preservedType =
            cloneMemRefWithMemorySpace(outRankedMemrefType, outMemSpace);
        auto castedMemref = memref::CastOp::create(
            rewriter, op->getLoc(), preservedType, newRankedMemref);
        rewriter.replaceAllUsesWith(output, castedMemref);
      }

      rewriter.eraseOp(op);
      return success();
    }
    if (isa<triton::PointerType, ptr::PtrType>(inType) &&
        outUnrankedMemrefType) {
      // Derive the memory space from the input ptr type (same reasoning as
      // the ranked branch above).
      Attribute outMemSpace = getMemorySpaceForPtr(inType);

      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = getMemrefElementTypeForPtrCast(
          inType, outUnrankedMemrefType.getElementType());
      // BUGFIX: Use dynamic size for unranked output (masked_select bug fix)
      auto ptrToMemrefType = MemRefType::get({ShapedType::kDynamic}, elemType,
                                             AffineMap(), outMemSpace);
      auto ptrToMemref = ptr::FromPtrOp::create(
          rewriter, op->getLoc(), ptrToMemrefType, input, Value());

      // BUGFIX: Use large size instead of 1
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(0x7FFFFFFF)};
      SmallVector<OpFoldResult> newStrides = {rewriter.getIndexAttr(1)};
      auto rankedDynamicMemrefType = MemRefType::get(
          {ShapedType::kDynamic}, elemType, AffineMap(), outMemSpace);
      auto newRankedMemref = memref::ReinterpretCastOp::create(
          rewriter, op->getLoc(), rankedDynamicMemrefType, ptrToMemref,
          rewriter.getIndexAttr(0), sizes, newStrides);

      if (output.getType() == newRankedMemref.getType()) {
        rewriter.replaceAllUsesWith(output, newRankedMemref);
      } else {
        auto convertedType = dyn_cast<MemRefType>(output.getType());
        if (!convertedType) {
          op.emitError() << "expected unrealized ptr->memref cast result to be "
                            "converted to ranked memref, got: "
                         << output.getType();
          return failure();
        }
        auto preservedType =
            cloneMemRefWithMemorySpace(convertedType, outMemSpace);
        auto castedMemref = memref::CastOp::create(
            rewriter, op->getLoc(), preservedType, newRankedMemref);
        rewriter.replaceAllUsesWith(output, castedMemref);
      }

      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class ReconcilePtrCastsPass
    : public triton::impl::ReconcilePtrCastsBase<ReconcilePtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ptr::PtrDialect, memref::MemRefDialect, BuiltinDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MemrefCastConverter, SimplifyUnrealizedCast,
                 FromMemrefConverter, ToMemrefConverter>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createReconcilePtrCastsPass() {
  return std::make_unique<ReconcilePtrCastsPass>();
}
