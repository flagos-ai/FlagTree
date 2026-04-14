#ifndef TLE_RAW_CONVERSION_TLETOLLVM_EXCLUSIVECUMSUMOPTOLLVM_H
#define TLE_RAW_CONVERSION_TLETOLLVM_EXCLUSIVECUMSUMOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::tle {
void populateExclusiveCumsumOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);
}

#endif
