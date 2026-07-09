#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"

using namespace mlir;
namespace tle = mlir::triton::tle;

namespace {

struct TMAStoreCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<tle::TMAStoreCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::TMAStoreCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<NVVM::CpAsyncBulkCommitGroupOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void tle::populateTMAStoreCommitGroupOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<TMAStoreCommitGroupOpConversion>(typeConverter, benefit);
}
