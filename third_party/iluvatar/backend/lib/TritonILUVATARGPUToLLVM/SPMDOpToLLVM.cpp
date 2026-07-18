#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        ::mlir::gpu::GridDimOp::create(rewriter, loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

} // namespace

void mlir::triton::ILUVATAR::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  // patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
}
