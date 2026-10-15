#include "Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace tle = mlir::triton::iluvatar_tle;

Value getDistDevicePtr(tle::GetDeviceIdOp op, SmallVector<Value> &srcElems) {
  if (!srcElems.empty())
    return srcElems[0];
  else {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    // arg0: memory pointer
    // arg1: communicator pointer
    return func.getArgument(1);
  }
}

struct GetDeviceIdOpConversion
    : public ConvertOpToLLVMPattern<tle::GetDeviceIdOp> {
  GetDeviceIdOpConversion(LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::GetDeviceIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> srcElems;
    if (auto src = adaptor.getInput())
      srcElems = unpackLLElements(loc, src, rewriter);
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      return rewriter.notifyMatchFailure(
          op, "expected parent LLVM::LLVMFuncOp, but none was found. ");
    }
    auto comm = getDistDevicePtr(op, srcElems);
    rewriter.modifyOpInPlace(op, [&]() { op->insertOperands(0, comm); });
    auto localRank = rewriter.create<tle::GetLocalRankOp>(
        op.getLoc(), rewriter.getI32Type(), comm);
    rewriter.replaceOp(op, localRank.getResult());

    return success();
  }
};

} // namespace

void mlir::triton::iluvatar_tle::populateGetDeviceIdOpToFlagCxPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetDeviceIdOpConversion>(typeConverter, benefit);
}
