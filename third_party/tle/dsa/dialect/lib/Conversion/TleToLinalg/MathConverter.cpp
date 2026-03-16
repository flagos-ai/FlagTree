// Copyright 2026- Xcoresigma Technology Co., Ltd
#include "tle/dsa/dialect/include/Conversion/TleToLinalg/MathConverter.h"
#include "tle/dsa/dialect/include/IR/Dialect.h"

namespace TleMathConverter {

using namespace mlir;
using namespace triton::tle;

} // namespace TleMathConverter

namespace mlir::triton::tle {
void populateTleMathOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSAAddOp,
                                                     hivm::VAddOp>>(
      patterns.getContext());
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSASubOp,
                                                     hivm::VSubOp>>(
      patterns.getContext());
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSAMulOp,
                                                     hivm::VMulOp>>(
      patterns.getContext());
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSADivOp,
                                                     hivm::VDivOp>>(
      patterns.getContext());
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSAMaxOp,
                                                     hivm::VMaxOp>>(
      patterns.getContext());
  patterns.add<TleMathConverter::BinaryMathConverter<triton::tle::DSAMinOp,
                                                     hivm::VMinOp>>(
      patterns.getContext());

  /// patterns.add<TleMathConverter::MatMulConverter<triton::tle::DSADotOp>>(patterns.getContext());
}
} // namespace mlir::triton::tle
