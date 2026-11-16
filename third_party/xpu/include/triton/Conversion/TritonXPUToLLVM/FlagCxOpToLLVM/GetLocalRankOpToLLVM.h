#ifndef TRITON_XPU_GET_LOCAL_RANK_OP_TO_LLVM_H_
#define TRITON_XPU_GET_LOCAL_RANK_OP_TO_LLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::xpu {

void populateGetLocalRankOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit);
void populateGetNumPesOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);

} // namespace mlir::triton::xpu

#endif // TRITON_XPU_GET_LOCAL_RANK_OP_TO_LLVM_H_
