#ifndef BRIDGE_UTILS_H
#define BRIDGE_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace bridge_utils {

/// Return the identity (or initial value) attribute for the reduce operation.
/// The identity is computed by looking at the operation with the reduce region
/// `combineOp` and based on the reduce return type `retType`.
FailureOr<ArrayAttr>
getIdentitiesFromCombineOp(Region &combineOp, ArrayRef<Type> retType,
                           ConversionPatternRewriter &rewriter);

class CudaTileTypeConverter : public TypeConverter {
public:
  CudaTileTypeConverter();
};

bool canMapToCudaTile(triton::FuncOp op, CudaTileTypeConverter &typeConverter);

enum class Signedness { None, Signed, Unsigned };
enum class IntegerUpCast { None, To_I16 };

Value upCastOrSelf(OpBuilder &builder, Location loc, Value input,
                   Signedness signedness, IntegerUpCast integerUpCast);

Value downCastOrSelf(
    OpBuilder &builder, Location loc, ArrayRef<Value> operands, Type retType,
    llvm::function_ref<Value(OpBuilder &, Location, Type, ArrayRef<Value>)>
        createOp,
    IntegerUpCast integerUpCast);

LogicalResult matchAndRewriteGenericOpImpl(
    Operation *op, ValueRange operands, const TypeConverter *converter,
    ConversionPatternRewriter &rewriter,
    llvm::function_ref<Value(OpBuilder &, Location, Type, ArrayRef<Value>)>
        createOp,
    Signedness signedness, IntegerUpCast integerUpCast);

template <class TritonOp, class CudaTileOp, Signedness signedness,
          IntegerUpCast integerUpCast>
class ConvertGenericOp : public OpConversionPattern<TritonOp> {
private:
  bool approxModifier = false;
  bool flushToZeroModifier = false;

public:
  ConvertGenericOp(TypeConverter &typeConverter, MLIRContext *context,
                   bool approx = false, bool flushToZero = false)
      : OpConversionPattern<TritonOp>(typeConverter, context),
        approxModifier(approx), flushToZeroModifier(flushToZero) {}

  LogicalResult
  matchAndRewrite(TritonOp op, typename TritonOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return matchAndRewriteGenericOpImpl(
        op, adaptor.getOperands(), this->getTypeConverter(), rewriter,
        [&](OpBuilder &builder, Location loc, Type type,
            ArrayRef<Value> operands) {
          // For DivSIOp and RemSIOp, triton assume the LHS is positive in axis
          // analysis pass, see
          // https://github.com/triton-lang/triton/issues/7749. tileir backend
          // also assume the LHS is positive here for simplicity of the axis
          // analysis pass.
          // TODO: write a more general pass to analyze the all positive value.
          if constexpr (std::is_same_v<TritonOp, arith::DivSIOp>) {
            auto lhs = cuda_tile::AssumeOp::create(
                           rewriter, loc, operands[0],
                           cuda_tile::BoundedAttr::get(rewriter.getContext(), 0,
                                                       std::nullopt))
                           .getResult();
            auto tileSignedness = signedness == Signedness::Unsigned
                                      ? cuda_tile::Signedness::Unsigned
                                      : cuda_tile::Signedness::Signed;
            return CudaTileOp::create(builder, loc, type, lhs, operands[1],
                                      tileSignedness);
          } else if constexpr (std::is_same_v<TritonOp, arith::RemSIOp>) {
            auto tileSignedness = signedness == Signedness::Unsigned
                                      ? cuda_tile::Signedness::Unsigned
                                      : cuda_tile::Signedness::Signed;
            auto lhs = cuda_tile::AssumeOp::create(
                           rewriter, loc, operands[0],
                           cuda_tile::BoundedAttr::get(rewriter.getContext(), 0,
                                                       std::nullopt))
                           .getResult();
            return CudaTileOp::create(builder, loc, type, lhs, operands[1],
                                      tileSignedness);
          } else if constexpr (std::is_same_v<TritonOp, arith::DivSIOp> ||
                               std::is_same_v<TritonOp, arith::DivUIOp> ||
                               std::is_same_v<TritonOp, arith::CeilDivSIOp> ||
                               std::is_same_v<TritonOp, arith::CeilDivUIOp> ||
                               std::is_same_v<TritonOp, arith::FloorDivSIOp>) {
            assert(operands.size() == 2 && "expect two operands for divi");
            auto rounding = cuda_tile::RoundingMode::ZERO;
            if (std::is_same_v<TritonOp, arith::CeilDivSIOp> ||
                std::is_same_v<TritonOp, arith::CeilDivUIOp>)
              rounding = cuda_tile::RoundingMode::POSITIVE_INF;
            else if (std::is_same_v<TritonOp, arith::FloorDivSIOp>)
              rounding = cuda_tile::RoundingMode::NEGATIVE_INF;
            auto tileSignedness = signedness == Signedness::Unsigned
                                      ? cuda_tile::Signedness::Unsigned
                                      : cuda_tile::Signedness::Signed;
            return CudaTileOp::create(builder, loc, type, operands[0],
                                      operands[1], tileSignedness, rounding);
          } else if constexpr (std::is_same_v<TritonOp, arith::AddFOp> ||
                               std::is_same_v<TritonOp, arith::MulFOp> ||
                               std::is_same_v<TritonOp, arith::SubFOp>) {
            assert(operands.size() == 2 &&
                   "expect two operands for add/mul/sub");
            bool isF32 = getElementTypeOrSelf(op.getResult().getType()).isF32();
            bool ftzModifier = (this->flushToZeroModifier && isF32);
            return CudaTileOp::create(
                builder, loc, type, operands[0], operands[1],
                cuda_tile::RoundingMode::NEAREST_EVEN, ftzModifier);
          } else if constexpr (std::is_same_v<TritonOp, math::FmaOp>) {
            assert(operands.size() == 3 && "expect two operands for fma");
            bool isF32 = getElementTypeOrSelf(op.getResult().getType()).isF32();
            bool ftzModifier = (this->flushToZeroModifier && isF32);
            return CudaTileOp::create(
                builder, loc, type, operands[0], operands[1], operands[2],
                cuda_tile::RoundingMode::NEAREST_EVEN, ftzModifier);
          } else if constexpr (std::is_same_v<TritonOp, arith::DivFOp>) {
            assert(operands.size() == 2 && "expect two operands for div");
            bool isF32 = getElementTypeOrSelf(op.getResult().getType()).isF32();
            auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
            if (isF32) {
              rounding = cuda_tile::RoundingMode::FULL;
              if (this->approxModifier)
                rounding = cuda_tile::RoundingMode::APPROX;
            }
            bool ftzModifier = (this->flushToZeroModifier && isF32);
            return CudaTileOp::create(builder, loc, operands[0], operands[1],
                                      rounding, ftzModifier);
          } else if constexpr (std::is_same_v<TritonOp,
                                              triton::PreciseDivFOp>) {
            // Lower a precise div operation. The ftz flag will not
            // have any effect.
            return CudaTileOp::create(builder, loc, operands[0], operands[1],
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
          } else if constexpr (std::is_same_v<TritonOp, math::ExpOp>) {
            assert(operands.size() == 1 && "expect single operand for exp");
            return CudaTileOp::create(builder, loc, operands[0]);
          } else if constexpr (std::is_same_v<TritonOp, math::Exp2Op>) {
            assert(operands.size() == 1 && "expect single operand for ex2");
            bool isF32 = getElementTypeOrSelf(op.getResult().getType()).isF32();
            bool ftzModifier = (this->flushToZeroModifier && isF32);
            return CudaTileOp::create(builder, loc, operands[0], ftzModifier);
          } else if constexpr (std::is_same_v<TritonOp, math::SqrtOp>) {
            assert(operands.size() == 1 && "expect single operand for sqrt");
            bool isF32 = getElementTypeOrSelf(op.getResult().getType()).isF32();
            auto rounding = cuda_tile::RoundingMode::NEAREST_EVEN;
            if (this->approxModifier && isF32)
              rounding = cuda_tile::RoundingMode::APPROX;
            bool ftzModifier = (this->flushToZeroModifier && isF32);
            return CudaTileOp::create(builder, loc, operands[0], rounding,
                                      ftzModifier);
          } else if constexpr (std::is_same_v<TritonOp,
                                              triton::PreciseSqrtOp>) {
            // Lower a precise sqrt operation. The ftz flag will not
            // have any effect.
            return CudaTileOp::create(builder, loc, operands[0],
                                      cuda_tile::RoundingMode::NEAREST_EVEN);
          } else if constexpr (signedness != Signedness::None) {
            auto tileSignedness = signedness == Signedness::Unsigned
                                      ? cuda_tile::Signedness::Unsigned
                                      : cuda_tile::Signedness::Signed;
            return CudaTileOp::create(builder, loc, type, operands,
                                      tileSignedness);
          } else {
            return CudaTileOp::create(builder, loc, type, operands);
          }
        },
        signedness, integerUpCast);
  }
};

} // end namespace bridge_utils
} // namespace mlir

#endif // BRIDGE_UTILS_H
