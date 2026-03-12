#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "flagtree/Common/UnifiedHardware.h"

#include "mlir-ext/Dialect/MathExt/IR/MathExt.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns_FlagTree.hpp"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace triton;

namespace {

// FIXME: There is no triton::BarrierOp currently.
struct BarrierConverter : public OpConversionPattern<mlir::gpu::BarrierOp> {
  using OpConversionPattern<mlir::gpu::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    rewriter.create<mk::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

// Similar with triton-cpu.
struct PrintOpConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // If the op has no operands, we can just print the prefix.
    if (op.getNumOperands() == 0) {
      rewriter.create<mk::PrintOp>(loc, TypeRange{}, op.getPrefix(),
                                   op.getHex(), ValueRange{},
                                   llvm::SmallVector<int, 0>{});
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = op.getOperands()[i];
      auto isSigned = {op.getIsSigned()[i]};
      // If the operand is not a ranked tensor, we should create a new tensor.
      // See mlir/lib/Interfaces/DestinationStyleOpInterface.cpp#L39
      if (!isa<RankedTensorType>(operand.getType())) {
        // NOTE: Use tensor.from_elements, the arith.constant will not translate
        // to linalg.fill
        auto emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, SmallVector<int64_t>{}, operand.getType());

        auto operandTensor = rewriter.create<tensor::InsertOp>(
            loc, operand, emptyTensor, ValueRange{});
        rewriter.create<mk::PrintOp>(loc, operandTensor.getType(),
                                     op.getPrefix(), op.getHex(),
                                     operandTensor.getResult(), isSigned);
        continue;
      }

      auto operandType = cast<RankedTensorType>(operand.getType());
      auto flattenTensor = operand;
      if (operandType.getRank() != 1) {
        SmallVector<int64_t> flatten_shape = {operandType.getNumElements()};
        auto targetType =
            RankedTensorType::get(flatten_shape, operandType.getElementType());
        // NOTE: Avoid to create global constant tensors
        SmallVector<ReassociationIndices> reassociation(1);
        for (unsigned i = 0; i < operandType.getRank(); ++i) {
          reassociation.front().push_back(i);
        }
        flattenTensor = rewriter.create<tensor::CollapseShapeOp>(
            loc, targetType, operand, reassociation);
      }

      rewriter.create<mk::PrintOp>(loc, flattenTensor.getType(), op.getPrefix(),
                                   op.getHex(), flattenTensor, isSigned);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct DotScaledConverter : public OpConversionPattern<triton::DotScaledOp> {
  using OpConversionPattern<triton::DotScaledOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get operands
    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getC();
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    auto aElemType = op.getAElemTypeAttr();
    auto bElemType = op.getBElemTypeAttr();
    auto fastMath = op.getFastMathAttr();

    // Get type information
    auto aType = a.getType();
    auto bType = b.getType();
    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();

    // Create initial zero tensor
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);
    TypedAttr constantAttr =
        static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), elementType, constantAttr);
    auto zeroes =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    // Perform scaled dot product operation
    Value res = rewriter
                    .create<mk::DotScaledOp>(loc, TypeRange{op.getType()}, a,
                                             aScale, b, bScale, zeroes,
                                             aElemType, bElemType, fastMath)
                    .getResult(0);

    // Check if C needs to be added
    bool skipC = isZeroTensor(c, false);
    if (!skipC) {
      res = rewriter.create<arith::AddFOp>(loc, c, res);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct GatherConverter : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern<triton::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value dstInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    auto gatherOp =
        rewriter.create<mk::GatherOp>(op.getLoc(), op.getType(), op.getSrc(),
                                      op.getIndices(), dstInit, op.getAxis());

    rewriter.replaceOp(op, gatherOp.getResult());
    return success();
  }
};

} // namespace

#endif
