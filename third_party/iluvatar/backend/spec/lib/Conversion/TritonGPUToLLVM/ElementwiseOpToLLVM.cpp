#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"

namespace {

// Prefer the simple LLVM exp2 intrinsic for f32 to avoid libdevice branching.
// For non-f32 inputs, fall back to the generic lowering.
struct Exp2OpConversion
    : mlir::triton::gpu::ElementwiseOpConversionBase<math::Exp2Op,
                                                     Exp2OpConversion> {
  using mlir::triton::gpu::ElementwiseOpConversionBase<
      math::Exp2Op, Exp2OpConversion>::ElementwiseOpConversionBase;

  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit = patternBenefitDefault)
      : mlir::triton::gpu::ElementwiseOpConversionBase<math::Exp2Op,
                                                       Exp2OpConversion>(
            typeConverter, axisAnalysisPass, benefit) {}

  llvm::SmallVector<mlir::Value>
  createDestOps(math::Exp2Op op, OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter, mlir::Type elemTy,
                mlir::triton::gpu::MultipleOperandsRange operands,
                mlir::Location loc) const {
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    llvm::StringRef funcName = "llvm.exp2.f32";
    mlir::Type funcType =
        mlir::triton::gpu::getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }
};

} // namespace

namespace mlir::triton::ILUVATAR {

void populateExp2OpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit) {
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis,
                                 PatternBenefit(benefit.getBenefit() + 1));
}

} // namespace mlir::triton::ILUVATAR

namespace mlir::triton::gpu {

SmallVector<Value> reorderValues(const SmallVector<Value> &values, Type inType,
                                 Type ouType) {
  return values;
}

SmallVector<Value> unpackI32(const SmallVector<Value> &inValues, Type srcTy,
                             ConversionPatternRewriter &rewriter, Location loc,
                             const LLVMTypeConverter *typeConverter) {
  return inValues;
}

SmallVector<Value> packI32(const SmallVector<Value> &inValues, Type srcTy,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter) {
  return inValues;
}

bool maybeDeduplicate_baseEncoding(Attribute baseEncoding) {
  if (isa<IluvatarMmaEncodingAttr, DotOperandEncodingAttr>(baseEncoding)) {
    // TODO: this logic seems incorrect for mma layout. Skip for now.
    // The following test crashes and some other miscompile:
    // test_core::test_fp8_dot_acc
    return true;
  }
  return false;
}

void matchAndRewrite_elemTy(const mlir::TypeConverter *typeConverter,
                            mlir::Type &elemTy, const mlir::Type &resultTy) {
  auto srcType = typeConverter->convertType(resultTy);
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(srcType))
    elemTy = structTy.getBody()[0];
}

} // namespace mlir::triton::gpu
