#ifndef ILUVATAR_TLE_CONVERSION_TLETOLLVMPASSES_EXTRACTOPTOLLVM_H
#define ILUVATAR_TLE_CONVERSION_TLETOLLVMPASSES_EXTRACTOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::iluvatar_tle {
void populateExtractOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);
}

#endif
