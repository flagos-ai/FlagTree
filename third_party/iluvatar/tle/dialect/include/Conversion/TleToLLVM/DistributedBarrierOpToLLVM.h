#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_DISTRIBUTEDBARRIEROPTOLLVM_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_DISTRIBUTEDBARRIEROPTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::iluvatar_tle {

void populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_DISTRIBUTEDBARRIEROPTOLLVM_H_
