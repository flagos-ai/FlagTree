#include "PatternPopulation.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace Intellifusion {

struct ConstantOpConversion : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType) {
      llvm::errs() << "[ConstantOpConversion] Not a tensor type\n";
      return failure();
    }

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr) {
      llvm::errs() << "[ConstantOpConversion] Not a dense elements attr\n";
      return failure();
    }

    auto loc = op.getLoc();
    auto structType = dyn_cast<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(tensorType));

    if (!structType) {
      llvm::errs()
          << "[ConstantOpConversion] Failed to convert to struct type\n";
      return failure();
    }

    Value constVal;
    if (denseAttr.isSplat()) {
      auto splat = denseAttr.getSplatValue<Attribute>();
      if (auto intAttr = dyn_cast<IntegerAttr>(splat)) {
        constVal = LLVM::ConstantOp::create(rewriter, loc,
                                            structType.getBody()[0], intAttr);
      } else if (auto floatAttr = dyn_cast<FloatAttr>(splat)) {
        constVal = LLVM::ConstantOp::create(rewriter, loc,
                                            structType.getBody()[0], floatAttr);
      } else {
        llvm::errs() << "[ConstantOpConversion] Unknown attribute type\n";
        return failure();
      }
    } else {
      llvm::errs() << "[ConstantOpConversion] Not a splat\n";
      return failure();
    }

    Value bitcast = LLVM::BitcastOp::create(rewriter, loc,
                                            structType.getBody()[0], constVal);

    Value undef = LLVM::UndefOp::create(rewriter, loc, structType);
    Value result = LLVM::InsertValueOp::create(rewriter, loc, undef, bitcast,
                                               ArrayRef<int64_t>{0});

    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateGPNPUArithToLLVMPatterns(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter) {
  patterns.add<ConstantOpConversion>(typeConverter);
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
