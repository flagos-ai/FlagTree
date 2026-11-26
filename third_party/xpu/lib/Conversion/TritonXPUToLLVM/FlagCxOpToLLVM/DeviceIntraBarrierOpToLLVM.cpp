#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"

#include "FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

using namespace mlir;

namespace {

using triton::xpu::createFlagcxDeviceCall;
using triton::xpu::FlagcxDeviceFunction;

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto function =
        llvm::StringSwitch<std::optional<FlagcxDeviceFunction>>(
            op.getKindAttr().getValue())
            .Case("sync", FlagcxDeviceFunction::IntraBarrierSync)
            .Case("arrive", FlagcxDeviceFunction::IntraBarrierArrive)
            .Case("wait", FlagcxDeviceFunction::IntraBarrierWait)
            .Default(std::nullopt);
    if (!function)
      return op.emitOpError("unsupported device barrier kind");

    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto i1Ty = rewriter.getI1Type();

    auto constant = [&](Type ty, int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, ty, rewriter.getIntegerAttr(ty, value));
    };
    Value coopKind = constant(i32Ty, op.getCoopKindAttr().getInt());
    Value barrierIndex = constant(i32Ty, op.getBarrierIndexAttr().getInt());
    Value multimem = constant(i1Ty, 0);
    Value order = constant(i32Ty, op.getOrderAttr().getInt());
    auto call = createFlagcxDeviceCall(op, rewriter, *function, adaptor.getComm(),
                                     {coopKind, barrierIndex, multimem, order});
    if (failed(call))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
