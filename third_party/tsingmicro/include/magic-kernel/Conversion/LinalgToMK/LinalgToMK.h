//===------------------- LinalgToMK.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering all linalg ops into mk ops.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_LINALG_TO_MK_H
#define ZTC_CONVERSION_LINALG_TO_MK_H

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

void populateLinalgToMKCanonicalizationPatterns(RewritePatternSet &patterns);

void populateLinalgToMKConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToMKPass();

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

// Extract the operations from a linalg op region
template <typename T> static bool checkGenericOp(linalg::GenericOp op) {
  auto regionBlock = op.getBody();
  auto regionOps = llvm::map_to_vector(regionBlock->without_terminator(),
                                       [](Operation &op) { return &op; });

  return regionOps.size() == 1 && isa<T>(regionOps[0]);
}

static bool isConstantTensor(Value &v, double targetValue) {
  auto fillOp = dyn_cast<linalg::FillOp>(v.getDefiningOp());
  if (!fillOp) {
    return false;
  }

  auto fillValue = fillOp.getInputs()[0];
  auto constOp = fillValue.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
    return val.getValueAsDouble() == targetValue;
  }
  if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
    return val.getValue() == static_cast<int64_t>(targetValue);
  }

  return false;
}

// Check if the given value is a tensor filled with 0.
static bool isZeroTensor(Value &v) { return isConstantTensor(v, 0.0); }

// Check if the given value is a tensor filled with 1.
static bool isOneTensor(Value &v) { return isConstantTensor(v, 1.0); }

static bool matchSigmoid(linalg::GenericOp op, Value &input) {
  // 1. sub (0 - x = -x)
  // 2. exp (e^(-x))
  // 3. add (1 + e^(-x))
  // 4. div (1 / (1 + e(^-x)))
  // We match the sigmoid pattern from down to up.

  // 1. Match div first.
  if (!checkGenericOp<arith::DivFOp>(op)) {
    return false;
  }

  auto divLhs = op.getInputs()[0];
  if (!isOneTensor(divLhs)) {
    return false;
  }

  // 2. Match add.
  auto addResult = op.getInputs()[1];
  auto addGenericOp = addResult.getDefiningOp<linalg::GenericOp>();
  if (!addGenericOp || !checkGenericOp<arith::AddFOp>(addGenericOp)) {
    return false;
  }

  auto addLhs = addGenericOp.getInputs()[0];
  auto addRhs = addGenericOp.getInputs()[1];
  bool isAddLhsOne = isOneTensor(addLhs);
  bool isAddRhsOne = isOneTensor(addRhs);
  if (!isAddLhsOne && !isAddRhsOne) {
    return false;
  }

  // 3. Match exp.
  auto expResult = isAddLhsOne ? addRhs : addLhs;
  auto expGenericOp = expResult.getDefiningOp<linalg::GenericOp>();
  if (!expGenericOp || !checkGenericOp<math::ExpOp>(expGenericOp)) {
    return false;
  }

  // 4. Match sub.
  auto subResult = expGenericOp.getInputs()[0];
  auto subGenericOp = subResult.getDefiningOp<linalg::GenericOp>();
  if (!subGenericOp || !checkGenericOp<arith::SubFOp>(subGenericOp)) {
    return false;
  }

  auto subLhs = subGenericOp.getInputs()[0];
  if (!isZeroTensor(subLhs)) {
    return false;
  }

  // Set input of Sub operation to the input of the sigmoid op.
  input = subGenericOp.getInputs()[1];

  // Match sigmoid pattern successfully.
  return true;
}

struct SigmoidFusionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Match sigmoid pattern
    Location loc = op.getLoc();
    Value input;
    if (!matchSigmoid(op, input)) {
      return rewriter.notifyMatchFailure(op, "sigmoid pattern not matched");
    }

    auto dstType = cast<RankedTensorType>(op.getType(0));
    auto elementType = dstType.getElementType();
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);

    // Replace the div GenericOp with mk::SigmoidOp
    // We can use CSE to erase other unused generic ops.
    auto sigmoidOp = rewriter.replaceOpWithNewOp<mk::SigmoidOp>(
        op, dstType, input, init, rewriter.getBoolAttr(false));

    return success();
  }
};

} // namespace

#endif // ZTC_CONVERSION_MEMREF_TO_MAGICKERNEL_H
