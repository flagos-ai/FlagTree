#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

class ExternCallOpConversion
    : public ConvertOpToLLVMPattern<triton::ExternCallOp> {
public:
  ExternCallOpConversion(const LLVMTypeConverter &converter,
                         const PatternBenefit &benefit)
      : ConvertOpToLLVMPattern<triton::ExternCallOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ExternCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1) {
      llvm::errs() << "ExternCallConversion does not support multi outs.";
      return failure();
    }

    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    std::string funcName = op.getSymbol().str();
    StringRef libname = op.getLibname();
    StringRef libpath = op.getLibpath();

    Operation *externCallOp;
    Type funcType = mlir::triton::gpu::getFunctionType(retType, newOperands);
    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, libname, libpath);
    externCallOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, newOperands);

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, externCallOp->getResult(0));
    }

    return success();
  }
};

} // namespace

void mlir::triton::populateExternCallOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ExternCallOpConversion>(typeConverter, benefit);
}
