#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::HCU {

void populateExtractSliceOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

void populateInThreadTransposeOpToTTGPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::PatternBenefit benefit);
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit);

void populateScaledUpcastOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_
