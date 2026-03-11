#ifndef TLE_CONVERSION_TLE_TO_LLVM_DISTRIBUTED_BARRIER_OP_TO_LLVM_H
#define TLE_CONVERSION_TLE_TO_LLVM_DISTRIBUTED_BARRIER_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::tle {

void populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

} // namespace mlir::triton::tle

#endif
