/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "DiscreteMaskAccessConversion/Passes.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "TritonToLinalg/MaskAnalysis.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "ascend/triton-adapter/include/DiscreteMaskAccessConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

struct DiscreteMaskStoreConversion
    : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

LogicalResult matchAndRewrite(triton::StoreOp op,
                              PatternRewriter &rewriter) const final {
  auto mask = op.getMask();
  auto loc = op.getLoc();
  auto dst = op.getPtr();
  auto src = op.getValue();

  if (!mask)
    return failure();
  
  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }

  auto loadFromDstOp = rewriter.create<triton::LoadOp>(
      loc, dst, op.getCache(), op.getEvict(), false);

  auto selOp = rewriter.create<arith::SelectOp>(loc, mask, src, loadFromDstOp.getResult());
  auto newStore = rewriter.create<triton::StoreOp>(
              loc, dst, selOp, op.getCache(), op.getEvict());
  newStore->setAttr(ConverterUtils::discreteMaskAttrName, UnitAttr::get(rewriter.getContext()));
  rewriter.replaceOp(op, newStore);
  return success();
}
};

struct DiscreteMaskLoadConversion
    : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

LogicalResult matchAndRewrite(triton::LoadOp op,
                              PatternRewriter &rewriter) const final {
  auto loc = op.getLoc();
  auto other = op.getOther();
  auto mask = op.getMask();
  auto ptr = op.getPtr();

  if (!mask)
    return failure();
  
  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }

  if (!other) {
    auto ptrType = ptr.getType();
    auto elementType = getElementTypeOrSelf(ptrType); 
    if (auto intType = dyn_cast<IntegerType>(ptrType)) {
      other = rewriter.create<arith::ConstantOp>(
                      loc, elementType, rewriter.getIntegerAttr(elementType, 0));
    } else if (auto floatType = dyn_cast<FloatType>(ptrType)) {
      other = rewriter.create<arith::ConstantOp>(
                      loc, elementType, rewriter.getFloatAttr(elementType, 0.0));
    } else {
      llvm_unreachable("Unsupported type for constant creation");
    }
  }

  auto newLoadOp = rewriter.create<triton::LoadOp>(
      loc, ptr, op.getCache(), op.getEvict(), op.getIsVolatile());
  auto discreteMaskOp = rewriter.create<arith::SelectOp>(loc, mask, newLoadOp, other);
  rewriter.replaceOp(op, discreteMaskOp);
  return success();
}
};

struct DiscreteMaskAtomicAddConversion : OpRewritePattern<triton::AtomicRMWOp> {
using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
LogicalResult matchAndRewrite(triton::AtomicRMWOp op, PatternRewriter &rewriter) const final {
  if (op.getAtomicRmwOp() != triton::RMWOp::FADD && op.getAtomicRmwOp() != triton::RMWOp::ADD) {
    return failure();
  }
  auto loc = op.getLoc();
  auto ptr = op.getPtr();
  auto value = op.getVal();
  auto mask = op.getMask();

  if (!mask)
    return failure();

  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }

  mlir::Value zeros;
  auto valueType = value.getType();
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(valueType)) {
    auto elemType = tensorType.getElementType();
    auto zeroAttr = rewriter.getZeroAttr(elemType);
    auto denseAttr = mlir::DenseElementsAttr::get(tensorType, zeroAttr);
    zeros = rewriter.create<mlir::arith::ConstantOp>(loc, denseAttr);
  } else if (mlir::isa<mlir::FloatType>(valueType) || mlir::isa<mlir::IntegerType>(valueType)) {
    auto zeroAttr = rewriter.getZeroAttr(valueType);
    zeros = rewriter.create<mlir::arith::ConstantOp>(loc, zeroAttr);
  } else {
    op.emitError() << "Unsupported value type for select: " << valueType << "\n";
    return failure();
  }
  auto maskedValue = rewriter.create<arith::SelectOp>(loc, mask, value, zeros);
  auto newAtomicAddOp = rewriter.create<triton::AtomicRMWOp>(
      loc, value.getType(), op.getAtomicRmwOp(), ptr, maskedValue, mlir::Value(), op.getSem(), op.getScope());
  rewriter.replaceOp(op, newAtomicAddOp);
  return success();
}
};

void DiscreteMaskAccessConversionPass::runOnOperation() {
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<DiscreteMaskLoadConversion>(patterns.getContext());
  patterns.add<DiscreteMaskStoreConversion>(patterns.getContext());
  patterns.add<DiscreteMaskAtomicAddConversion>(patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply discrete mask access patterns");
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createDiscreteMaskAccessConversionPass() {
  return std::make_unique<DiscreteMaskAccessConversionPass>();
}