#include "tle/dialect/include/Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"
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

struct GetDeviceIdOpConversion
    : public ConvertOpToLLVMPattern<tle::GetDeviceIdOp> {
  GetDeviceIdOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetDeviceIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getInput();

    auto localRank = rewriter.create<tle::GetLocalRankOp>(
        op.getLoc(), rewriter.getI32Type(), src);

    rewriter.replaceOp(op, localRank.getResult());

    return success();
  }
};

} // namespace

void tle::populateGetDeviceIdOpToFlagCxPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetDeviceIdOpConversion>(typeConverter, benefit);
}
