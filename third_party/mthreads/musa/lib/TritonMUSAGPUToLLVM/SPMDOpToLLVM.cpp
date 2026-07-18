#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef intrinsic;
    switch (op.getAxis()) {
    case ProgramIDDim::X:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.x";
      break;
    case ProgramIDDim::Y:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.y";
      break;
    case ProgramIDDim::Z:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.z";
      break;
    default:
      intrinsic = "llvm.musa.read.ptx.sreg.nctaid.x";
      break;
    }

    auto call =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, {});
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateSPMDOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
