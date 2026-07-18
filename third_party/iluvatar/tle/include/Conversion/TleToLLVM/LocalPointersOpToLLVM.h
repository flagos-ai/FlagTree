#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_LOCALPOINTERSOPTOLLVM_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_LOCALPOINTERSOPTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::iluvatar_tle {

void populateLocalPointersOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfoBase &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_LOCALPOINTERSOPTOLLVM_H_
