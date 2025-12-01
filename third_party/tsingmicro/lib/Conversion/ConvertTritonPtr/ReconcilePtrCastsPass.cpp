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

#include "Address/Dialect/IR/AddressDialect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/ConvertTritonPtr/ReconcilePtrCasts.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h.inc"

namespace {

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}

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

    auto unrealizedCast = in.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealizedCast)
      return failure();
    if (!isOneToOneCast(unrealizedCast)) {
      return failure();
    }

    if (!isa<triton::PointerType>(unrealizedCast.getType(0)))
      return failure();
    auto prevInput = unrealizedCast.getInputs().front();
    auto newCast = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

    rewriter.replaceOp(op, newCast);
    return success();
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

    if (unrankedInput && isa<addr::AddressType>(outType)) {
      // from_memref only takes ranked memref, cast the unranked memref to
      // ranked memref first.
      auto rankedMemref = rewriter.create<memref::CastOp>(
          op.getLoc(), MemRefType::get({1}, unrankedInput.getElementType()),
          input);
      auto memrefToPtr = rewriter.create<addr::FromMemRefOp>(
          op->getLoc(), addr::AddressType::get(rewriter.getContext()),
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
    auto outUnrankedMemrefType = dyn_cast<UnrankedMemRefType>(output.getType());
    if (isa<addr::AddressType>(inType) && outUnrankedMemrefType) {
      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = outUnrankedMemrefType.getElementType();
      auto ptrToMemref = rewriter.create<addr::ToMemRefOp>(
          op->getLoc(), MemRefType::get({1}, elemType), input);

      auto newUnrankedMemref = rewriter.create<memref::CastOp>(
          op.getLoc(), MemRefType::get({ShapedType::kDynamic}, elemType),
          ptrToMemref);

      rewriter.replaceAllUsesWith(output, newUnrankedMemref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class ReconcilePtrCastsPass
    : public ReconcilePtrCastsBase<ReconcilePtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<addr::AddressDialect, memref::MemRefDialect, BuiltinDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<SimplifyUnrealizedCast, FromMemrefConverter, ToMemrefConverter>(
            &getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createReconcilePtrCastsPass() {
  return std::make_unique<ReconcilePtrCastsPass>();
}
