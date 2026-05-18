#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ThreadIdOpPattern : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
public:
  using ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef intrinsic;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.x";
      break;
    case mlir::gpu::Dimension::y:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.y";
      break;
    case mlir::gpu::Dimension::z:
      intrinsic = "llvm.musa.read.ptx.sreg.tid.z";
      break;
    }

    Type ty = getTypeConverter()->convertType(op.getType());
    auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(),
                                                intrinsic, ty, {});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateThreadIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ThreadIdOpPattern>(typeConverter, benefit);
}
