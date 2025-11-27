#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"

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
