#ifndef TLE_RAW_CONVERSION_TLETOLLVM_LOCALPOINTERSOPTOLLVM_H
#define TLE_RAW_CONVERSION_TLETOLLVM_LOCALPOINTERSOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::tle {
void populateLocalPointersOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                           const TargetInfoBase &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);
}

#endif
