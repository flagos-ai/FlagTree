#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class WarpIdOpPattern
    : public ConvertOpToLLVMPattern<mlir::triton::gpu::WarpIdOp> {
public:
  using ConvertOpToLLVMPattern<
      mlir::triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // This is runtime-constant for a program instance; move it to function
    // entry unless we are inside a warp-specialized partition.
    std::optional<int> startWarpId = getWarpGroupStartWarpId(op->getBlock());
    if (!startWarpId) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      rewriter.setInsertionPoint(
          &funcOp.getFunctionBody().getBlocks().front().front());
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tid = LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.musa.read.ptx.sreg.tid.x", i32_ty, {})
                    .getResult(0);
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpId = b.udiv(tid, b.i32_val(threadsPerWarp));

    if (startWarpId)
      warpId = b.sub(warpId, b.i32_val(*startWarpId));

    rewriter.replaceOp(op, warpId);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, benefit);
}
