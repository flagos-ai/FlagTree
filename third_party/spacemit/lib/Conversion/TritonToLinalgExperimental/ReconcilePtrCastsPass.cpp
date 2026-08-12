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
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"

using namespace mlir;
using namespace triton;

static ptr::MemorySpaceAttrInterface
getMemorySpaceForMemref(BaseMemRefType memrefType) {
  if (auto memorySpace = dyn_cast_if_present<ptr::MemorySpaceAttrInterface>(
          memrefType.getMemorySpace()))
    return memorySpace;
  return ptr::GenericSpaceAttr::get(memrefType.getContext());
}

static ptr::PtrType getPtrTypeForMemref(BaseMemRefType memrefType) {
  return ptr::PtrType::get(memrefType.getContext(),
                           getMemorySpaceForMemref(memrefType));
}

static MemRefType getRankedMemrefTypeForPtrCast(BaseMemRefType memrefType) {
  return MemRefType::get({1}, memrefType.getElementType(), AffineMap(),
                         getMemorySpaceForMemref(memrefType));
}

static MemRefType cloneMemRefWithMemorySpace(MemRefType memrefType,
                                             Attribute memorySpace) {
  return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                         memrefType.getLayout(), memorySpace);
}

static ptr::MemorySpaceAttrInterface getMemorySpaceForPtr(Type ptrType) {
  if (auto ptrPtrType = dyn_cast<ptr::PtrType>(ptrType))
    return ptrPtrType.getMemorySpace();
  return ptr::GenericSpaceAttr::get(ptrType.getContext());
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
        if (!memref::CastOp::areCastCompatible(rankedInputTy, rankedResultTy))
          return failure();
        rewriter.replaceOpWithNewOp<memref::CastOp>(op, rankedResultTy, input);
        return success();
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
          auto rankedInput = memref::ReinterpretCastOp::create(
              rewriter, loc, rankedInputType, input, rewriter.getIndexAttr(0),
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
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
          // Cast unranked input to ranked memref<1 x srcElem>
          auto rankedInputType = getRankedMemrefTypeForPtrCast(unrankedInputTy);
          auto rankedInput = memref::ReinterpretCastOp::create(
              rewriter, loc, rankedInputType, input, rewriter.getIndexAttr(0),
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
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
      // ptr.to_ptr requires the ptr-like memory space and ptr memory space to
      // match. Missing memref memory space must be treated as an IR error.
      if (!unrankedInput.getMemorySpace()) {
        op.emitError()
            << "cannot lower unrealized memref->ptr cast without explicit "
               "memory space on memref type: "
            << unrankedInput;
        return failure();
      }

      // from_memref only takes ranked memref, cast the unranked memref to
      // ranked memref first.
      auto rankedType = getRankedMemrefTypeForPtrCast(unrankedInput);
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
      auto rankedMemref = memref::ReinterpretCastOp::create(
          rewriter, op.getLoc(), rankedType, input, rewriter.getIndexAttr(0),
          sizes, strides);
      auto memrefToPtr = ptr::ToPtrOp::create(
          rewriter, op->getLoc(), getPtrTypeForMemref(rankedMemref.getType()),
          rankedMemref);

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
      // ptr.from_ptr requires matching memory spaces. Missing target memref
      // memory space must be treated as an IR error.
      if (!outRankedMemrefType.getMemorySpace()) {
        op.emitError()
            << "cannot lower unrealized ptr->memref cast without explicit "
               "memory space on target memref type: "
            << outRankedMemrefType;
        return failure();
      }

      auto elemType = getMemrefElementTypeForPtrCast(
          inType, outRankedMemrefType.getElementType());
      Attribute outMemSpace = outRankedMemrefType.getMemorySpace();
      auto ptrToMemrefType =
          MemRefType::get({1}, elemType, AffineMap(), outMemSpace);
      auto ptrToMemref = ptr::FromPtrOp::create(
          rewriter, op->getLoc(), ptrToMemrefType, input, Value());

      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> newStrides;
      for (int64_t i = 0, e = outRankedMemrefType.getRank(); i < e; ++i) {
        sizes.push_back(
            ShapedType::isDynamic(outRankedMemrefType.getDimSize(i))
                ? rewriter.getIndexAttr(1)
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
      // ptr.from_ptr requires matching memory spaces. Missing target memref
      // memory space must be treated as an IR error.
      if (!outUnrankedMemrefType.getMemorySpace()) {
        op.emitError()
            << "cannot lower unrealized ptr->memref cast without explicit "
               "memory space on target memref type: "
            << outUnrankedMemrefType;
        return failure();
      }

      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = getMemrefElementTypeForPtrCast(
          inType, outUnrankedMemrefType.getElementType());
      Attribute outMemSpace = outUnrankedMemrefType.getMemorySpace();
      auto ptrToMemrefType =
          MemRefType::get({1}, elemType, AffineMap(), outMemSpace);
      auto ptrToMemref = ptr::FromPtrOp::create(
          rewriter, op->getLoc(), ptrToMemrefType, input, Value());

      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
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
