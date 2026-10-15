#include "Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"

#include "IR/Dialect.h"
#include "Tools/FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/raw_ostream.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace tle = mlir::triton::iluvatar_tle;

struct DeviceIntraBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DeviceIntraBarrierOp> {
  DeviceIntraBarrierOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::DeviceIntraBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto indexValue = op.getIndexAttr().getInt();
    auto coopValue = op.getCoopKindAttr().getValue();
    auto orderValue = op.getOrderAttr().getValue();
    auto barrierType = op.getBarrierTypeAttr().getValue();

    tle::getBarrierFuncCall(loc, rewriter, adaptor.getComm(), indexValue,
                            coopValue, orderValue, barrierType);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::iluvatar_tle::populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DeviceIntraBarrierOpConversion>(typeConverter, benefit);
}
