#include "PatternPopulation.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace Intellifusion {

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (getIntOrFloatOrPtrBitWidth(elemTy) != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    Type resultTy = operands[0][0].getType();
    StringRef name = "llvm.nvvm.ex2.approx.f32";
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, {prod});
    return {callOp.getResult(0)};
  }
};

// Lower fp32/fp64 rsqrt to the matching NVVM intrinsic. Other element types
// fall back to the standard LLVM lowering path.
struct RsqrtOpConversion
    : public ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion> {
  using Base = ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::RsqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto floatTy = dyn_cast<FloatType>(elemTy);
    if (!floatTy)
      return {};

    StringRef intrinsicName;
    switch (floatTy.getWidth()) {
    case 32:
      intrinsicName = "llvm.nvvm.rsqrt.approx.f";
      break;
    case 64:
      intrinsicName = "llvm.nvvm.rsqrt.approx.d";
      break;
    default:
      return {};
    }

    Value rsqrtVal = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, intrinsicName, elemTy, operands[0])
                         .getResult(0);
    return {rsqrtVal};
  }
};

void populateGPNPUMathToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit) {
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<math::ExpOp, math::ExpOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<math::Log2Op, math::Log2Op>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<math::SqrtOp, math::SqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<RsqrtOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  // Keep the generic rsqrt lowering as a fallback for element types we do not
  // map to a GPNPU-specific intrinsic yet.
  patterns.add<ElementwiseOpConversion<math::RsqrtOp, math::RsqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseToIntrinsicOpConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "llvm.nvvm.sqrt.rn.f", benefit);
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
