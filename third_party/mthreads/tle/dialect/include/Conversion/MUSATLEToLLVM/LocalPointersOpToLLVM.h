#ifndef MTHREADS_MUSATLE_CONVERSION_MUSATLETOLLVM_LOCALPOINTERSOPTOLLVM_H
#define MTHREADS_MUSATLE_CONVERSION_MUSATLETOLLVM_LOCALPOINTERSOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::musa_tle {

void populateMUSATLEToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

} // namespace mlir::triton::musa_tle

#endif // MTHREADS_MUSATLE_CONVERSION_MUSATLETOLLVM_LOCALPOINTERSOPTOLLVM_H
