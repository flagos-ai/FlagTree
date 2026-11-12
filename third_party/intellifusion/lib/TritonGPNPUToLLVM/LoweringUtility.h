#ifndef TRITON_INTELLIFUSION_GPNPU_LOWERING_UTILITY_H
#define TRITON_INTELLIFUSION_GPNPU_LOWERING_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::Intellifusion {

// Compute one shared-memory plan for both allocation and layout conversion.
std::pair<LinearLayout, std::pair<int32_t, int32_t>>
getLayoutConversionSwizzling(const LinearLayout &src, const LinearLayout &dst,
                             int32_t bitwidth,
                             const TargetInfoBase &targetInfo);

Value createElectPredicate(Location loc, OpBuilder &rewriter);
Value createElectPredicateWarp0(Location loc, OpBuilder &rewriter);
Value createTMAMulticastMask(Location loc, ConversionPatternRewriter &rewriter,
                             uint16_t broadcastBits);
Value createLeadCTAPredicate(Location loc, RewriterBase &rewriter);

} // namespace mlir::triton::Intellifusion

#endif // TRITON_INTELLIFUSION_GPNPU_LOWERING_UTILITY_H
