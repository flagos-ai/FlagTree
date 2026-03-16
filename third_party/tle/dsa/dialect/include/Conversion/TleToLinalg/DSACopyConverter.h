// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_CONVERSION_DSA_COPY_CONVERTER_H_
#define TRITON_TLE_CONVERSION_DSA_COPY_CONVERTER_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"

#include "tle/dsa/dialect/include/IR/Dialect.h"

namespace TleCopyConverter {

using namespace mlir;

class CopyConverter : public OpConversionPattern<triton::tle::DSACopyOp> {

public:
  explicit CopyConverter(MLIRContext *context);
  using OpConversionPattern<triton::tle::DSACopyOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::tle::DSACopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace TleCopyConverter

namespace mlir::triton::tle {
void populateTleCopyOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);
}

#endif
