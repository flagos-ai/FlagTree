#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

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
    
    auto coopValue = op.getCoopKindAttr().getInt();
    
    auto orderValue = op.getOrderAttr().getInt();
    
    auto barrierType = op.getBarrierTypeAttr().getValue();

    if (!llvm::is_contained(std::array<size_t, 5>{0, 1, 2, 3, 4}, coopValue))
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");

    if (!llvm::is_contained(std::array<size_t, 4>{0, 1, 2, 3}, orderValue))
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");
    
    llvm::errs() << "Lowering DeviceIntraBarrierOp to FlagCx function call\n";
    tle::getBarrierFuncCall(loc, rewriter, adaptor.getComm(), indexValue,
                            coopValue, orderValue, barrierType);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void tle::populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DeviceIntraBarrierOpConversion>(typeConverter, benefit);
}
