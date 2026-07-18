#include "tle/dialect/include/Conversion/TleToLLVM/GetLocalRankOpToLLVM.h"
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

struct GetNumPesOpConversion : public ConvertOpToLLVMPattern<tle::GetNumPesOp> {
  GetNumPesOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetNumPesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[GetNumPesOpConversion] " << msg << "\n";
      return failure();
    };
    auto loc = op.getLoc();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto getLocalPeCall = tle::getNumPesFunCall(loc, rewriter, srcElems[0]);

    Value nPes = getLocalPeCall.getResult();
    if (!nPes.getType().isInteger(32))
      return reportFailure("expected i32 result");
    rewriter.replaceOp(op, nPes);
    return success();
  }
};

struct GetLocalRankOpConversion
    : public ConvertOpToLLVMPattern<tle::GetLocalRankOp> {
  GetLocalRankOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetLocalRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[GetLocalRankOpConversion] " << msg << "\n";
      return failure();
    };
    auto loc = op.getLoc();
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto getLocalPeCall = tle::getLocalPeFuncCall(loc, rewriter, srcElems[0]);

    Value localPe = getLocalPeCall.getResult();
    if (!localPe.getType().isInteger(32))
      return reportFailure("expected i32 result");
    rewriter.replaceOp(op, localPe);

    return success();
  }
};

} // namespace
void tle::populateGetLocalRankOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               PatternBenefit benefit) {
  patterns.add<GetLocalRankOpConversion>(typeConverter, benefit);
}

void tle::populateGetNumPesOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<GetNumPesOpConversion>(typeConverter, benefit);
}
