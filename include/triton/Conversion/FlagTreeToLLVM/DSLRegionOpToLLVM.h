#ifndef FLAGTREE_CONVERSION_FLAGTREETOLLVMPASSES_DSLREGIONOPTOLLVM_H
#define FLAGTREE_CONVERSION_FLAGTREETOLLVMPASSES_DSLREGIONOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::flagtree {
void populateDSLRegionOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);
}

#endif
