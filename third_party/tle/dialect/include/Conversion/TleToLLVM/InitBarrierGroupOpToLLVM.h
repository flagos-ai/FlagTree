#ifndef TRITON_TLE_CONVERSION_INIT_BARRIER_GROUP_OP_TO_LLVM_H
#define TRITON_TLE_CONVERSION_INIT_BARRIER_GROUP_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::tle {

void populateInitBarrierGroupOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

} // namespace mlir::triton::tle

#endif
