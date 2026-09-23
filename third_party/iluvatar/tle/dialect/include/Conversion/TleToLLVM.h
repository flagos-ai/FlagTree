#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_TLETOLLVM_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_TLETOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::iluvatar_tle {

void populateTleToLLVMPatterns(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               RewritePatternSet &patterns,
                               PatternBenefit benefit);

void populateExtractTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         const TargetInfoBase &targetInfo,
                                         PatternBenefit benefit = 1);

void populateInsertTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit = 1);

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_TLETOLLVM_H_
