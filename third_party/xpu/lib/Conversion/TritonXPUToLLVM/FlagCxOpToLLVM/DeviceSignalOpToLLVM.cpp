#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceSignalOpToLLVM.h"

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

// The signal aggregate is a sum of 64-bit words, so the narrower compare has no
// meaning here; FlagCX's own callers pass 64 as well.
constexpr int32_t kSignalCompareBits = 64;

struct SignalOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();

    auto constant = [&](int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(value));
    };
    bool isAdd = op.getSignalKindAttr().getValue() == "add";
    auto function = isAdd ? FlagcxDeviceFunction::SignalAdd
                          : FlagcxDeviceFunction::SignalInc;
    SmallVector<Value> args{constant(op.getTeamKindAttr().getInt()),
                            adaptor.getPeer(), adaptor.getSlot()};
    if (isAdd)
      args.push_back(adaptor.getAddend());
    args.append({constant(op.getContextIdAttr().getInt()),
                 constant(op.getCoopKindAttr().getInt()),
                 constant(op.getScopeAttr().getInt())});
    if (failed(createFlagcxDeviceCall(op, rewriter, function, adaptor.getComm(),
                                     args)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct SignalWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto function =
        llvm::StringSwitch<std::optional<FlagcxDeviceFunction>>(
            op.getWaitKindAttr().getValue())
            .Case("signal", FlagcxDeviceFunction::WaitSignal)
            .Case("counter", FlagcxDeviceFunction::WaitCounter)
            .Case("shadow", FlagcxDeviceFunction::WaitSignalMeetShadow)
            .Default(std::nullopt);
    if (!function)
      return op.emitOpError("unsupported signal wait kind");

    auto loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();
    auto constant = [&](int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(value));
    };
    Value bits = constant(kSignalCompareBits);
    Value contextId = constant(op.getContextIdAttr().getInt());
    Value coopKind = constant(op.getCoopKindAttr().getInt());
    Value order = constant(op.getOrderAttr().getInt());

    SmallVector<Value> args;
    if (*function == FlagcxDeviceFunction::WaitSignalMeetShadow) {
      // MeetShadow takes the slot after the context and has no `least`.
      args = {contextId, adaptor.getSlot(), bits, coopKind, order};
    } else {
      args = {adaptor.getSlot(), adaptor.getLeast(), bits, contextId, coopKind,
              order};
    }
    if (failed(createFlagcxDeviceCall(op, rewriter, *function,
                                     adaptor.getComm(), args)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceSignalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SignalOpConversion, SignalWaitOpConversion>(typeConverter,
                                                           benefit);
}
