// Copyright 2026- Xcoresigma Technology Co., Ltd
#ifndef TRITON_TLE_CONVERSION_MATH_CONVERTER_H
#define TRITON_TLE_CONVERSION_MATH_CONVERTER_H

#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace TleMathConverter {

using namespace mlir;

template <typename MathOp, typename HivmOp>
class BinaryMathConverter : public OpConversionPattern<MathOp> {
public:
  using OpConversionPattern<MathOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = adaptor.getRes();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (result.getType() != lhs.getType() ||
        result.getType() != rhs.getType()) {
      op->emitError("Unexpected binary calculation type!");
      return failure();
    }

    auto binOp = rewriter.create<HivmOp>(loc, TypeRange{}, ValueRange{lhs, rhs},
                                         ValueRange{result});

    rewriter.replaceOp(op, binOp);
    return success();
  }
};

template <typename MathOp, typename HivmOp>
class UnaryMathConverter : public OpConversionPattern<MathOp> {
public:
  using OpConversionPattern<MathOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {}
};

template <typename MathOp>
class MatMulConverter : public OpConversionPattern<MathOp> {
public:
  static constexpr llvm::StringLiteral fixpipeAlreadyInserted =
      "fixpipe_already_inserted";
  using OpConversionPattern<MathOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MathOp op, typename MathOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inA = adaptor.getInA();
    auto inB = adaptor.getInB();
    auto res = adaptor.getRes();

    auto sizeAttr = adaptor.getSize();
    if (sizeAttr.size() > 3) {
      op->emitError("Unexpected matmul calculation size!");
      return failure();
    }

    auto mAttr = dyn_cast<IntegerAttr>(sizeAttr[0]);
    auto nAttr = dyn_cast<IntegerAttr>(sizeAttr[1]);
    auto kAttr = dyn_cast<IntegerAttr>(sizeAttr[2]);
    Value M = rewriter.create<arith::ConstantIndexOp>(loc, mAttr.getInt());
    Value N = rewriter.create<arith::ConstantIndexOp>(loc, nAttr.getInt());
    Value K = rewriter.create<arith::ConstantIndexOp>(loc, kAttr.getInt());

    bool initC = adaptor.getInitC();
    auto initCValue =
        rewriter.create<arith::ConstantIntOp>(loc,
                                              /*value*/ initC, /*width*/ 1);
    auto newOp = rewriter.create<hivm::MmadL1Op>(
        loc, TypeRange{}, // result types
        inA,              // Matrix A
        inB,              // Matrix B
        initCValue,       // init condition
        M,                // M
        K,                // K
        N,                // N
        res,              // Matrix C
        Value{},          // per channel bias
        adaptor.getTraA() ? rewriter.getUnitAttr() : UnitAttr{}, // transpose A
        adaptor.getTraB() ? rewriter.getUnitAttr() : UnitAttr{}, // transpose B
        adaptor.getEnableHf32() ? rewriter.getUnitAttr() : UnitAttr{}
        // enable hf32 mode
    );

    newOp->setAttr(fixpipeAlreadyInserted, rewriter.getBoolAttr(true));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

} // namespace TleMathConverter

namespace mlir::triton::tle {
void populateTleMathOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);
}
#endif
