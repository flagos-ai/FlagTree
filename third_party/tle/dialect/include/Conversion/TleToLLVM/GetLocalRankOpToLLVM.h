#ifndef TLE_CONVERSION_TLETOLLVM_GETLOCALPEOPTOLLVM_H
#define TLE_CONVERSION_TLETOLLVM_GETLOCALPEOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::tle {

void populateGetLocalRankOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit);
void populateGetNumPesOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);
} // namespace mlir::triton::tle

#endif
