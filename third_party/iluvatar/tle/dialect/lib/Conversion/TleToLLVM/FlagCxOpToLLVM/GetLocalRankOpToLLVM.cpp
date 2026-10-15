#include "Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"

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
    auto getNumPesCall = tle::getNumPesFunCall(loc, rewriter, srcElems[0]);

    Value nPes = getNumPesCall.getResult();
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
    auto comm = op.getSrc();
    auto getLocalPeCall = tle::getLocalPeFuncCall(loc, rewriter, comm);

    Value localPe = getLocalPeCall.getResult();
    if (!localPe.getType().isInteger(32))
      return reportFailure("expected i32 result");
    rewriter.replaceOp(op, localPe);
    return success();
  }
};

} // namespace

void mlir::triton::iluvatar_tle::populateGetLocalRankOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetLocalRankOpConversion>(typeConverter, benefit);
}

void mlir::triton::iluvatar_tle::populateGetNumPesOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumPesOpConversion>(typeConverter, benefit);
}
