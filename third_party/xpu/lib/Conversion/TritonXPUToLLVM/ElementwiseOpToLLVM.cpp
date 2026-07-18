#include "PatternTritonXPUOpToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public mlir::triton::gpu::ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {

  using Base = mlir::triton::gpu::ElementwiseOpConversionBase<
      SourceOp, ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp>
  createDestOps(SourceOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                mlir::triton::gpu::MultipleOperandsRange operands,
                Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public triton::gpu::ElementwiseOpConversionBase<
          TritonOp, OpToExternCallConversion<TritonOp>> {
  using Base = triton::gpu::ElementwiseOpConversionBase<
      TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy,
                                   triton::gpu::MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = triton::gpu::getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        triton::gpu::appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};

struct FpToFpOpConversion
    : public triton::gpu::ElementwiseOpConversionBase<triton::FpToFpOp,
                                                      FpToFpOpConversion> {
  using Base = triton::gpu::ElementwiseOpConversionBase<triton::FpToFpOp,
                                                        FpToFpOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value convertFloat8E4M3FNUZToFloat32(Value v,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Operation *op) const {
    std::string callFunc = "_ZN3xpu20fp8e4m3_fnuz_to_fp32Eh";
    ValueRange args{v};
    Type resultType = rewriter.getF32Type();
    return mlir::LLVM::XPU::createDeviceCall(callFunc, rewriter, op, resultType,
                                             args, loc);
  }

  Value convertFloat8E5M2FNUZToFloat32(Value v,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Operation *op) const {
    std::string callFunc = "_ZN3xpu20fp8e5m2_fnuz_to_fp32Eh";
    ValueRange args{v};
    Type resultType = rewriter.getF32Type();
    return mlir::LLVM::XPU::createDeviceCall(callFunc, rewriter, op, resultType,
                                             args, loc);
  }

  Value convertFloat32ToFloat8E4M3FNUZ_RTNE(Value v,
                                            ConversionPatternRewriter &rewriter,
                                            Location loc, Operation *op) const {
    std::string callFunc = "_ZN3xpu25fp32_to_fp8_e4m3_fnuz_rneEf";
    ValueRange args{v};
    Type resultType = rewriter.getI8Type();
    return mlir::LLVM::XPU::createDeviceCall(callFunc, rewriter, op, resultType,
                                             args, loc);
  }

  Value convertFloat32ToFloat8E5M2FNUZ_RTNE(Value v,
                                            ConversionPatternRewriter &rewriter,
                                            Location loc, Operation *op) const {
    std::string callFunc = "_ZN3xpu25fp32_to_fp8_e5m2_funz_rneEf";
    ValueRange args{v};
    Type resultType = rewriter.getI8Type();
    return mlir::LLVM::XPU::createDeviceCall(callFunc, rewriter, op, resultType,
                                             args, loc);
  }

  SmallVector<Value>
  createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                mlir::triton::gpu::MultipleOperandsRange operands,
                Location loc) const {
    auto getElementType = [&](Value v) {
      if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
        return tensorType.getElementType();
      }
      return v.getType();
    };

    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    // LLVM_DEBUG({
    // op->dump();
    // adaptor.getSrc().getType().dump();
    // dstElementType.dump();
    // });

    SmallVector<Value> outVals;

    if (isa<mlir::Float8E4M3FNUZType>(srcElementType) &&
        isa<mlir::Float32Type>(dstElementType)) {
      for (Value v : operands[0]) {
        outVals.push_back(convertFloat8E4M3FNUZToFloat32(v, rewriter, loc, op));
      }
      return outVals;
    } else if (isa<mlir::Float8E5M2FNUZType>(srcElementType) &&
               isa<mlir::Float32Type>(dstElementType)) {
      for (auto v : operands[0]) {
        outVals.push_back(convertFloat8E5M2FNUZToFloat32(v, rewriter, loc, op));
      }
      return outVals;
    } else if (isa<mlir::Float32Type>(srcElementType) &&
               isa<mlir::Float8E4M3FNUZType>(dstElementType)) {
      if (roundingMode.value_or(RoundingMode::RTNE) != RoundingMode::RTNE) {
        op->emitError("only support RoundingMode rtne\n");
      }
      for (auto v : operands[0]) {
        outVals.push_back(
            convertFloat32ToFloat8E4M3FNUZ_RTNE(v, rewriter, loc, op));
      }
      return outVals;
    } else if (isa<mlir::Float32Type>(srcElementType) &&
               isa<mlir::Float8E5M2FNUZType>(dstElementType)) {
      if (roundingMode.value_or(RoundingMode::RTNE) != RoundingMode::RTNE) {
        op->emitError("only support RoundingMode rtne\n");
      }
      for (auto v : operands[0]) {
        outVals.push_back(
            convertFloat32ToFloat8E5M2FNUZ_RTNE(v, rewriter, loc, op));
      }
      return outVals;
    }

    op.emitError("Unsupported source element type");
    return outVals;
  }
};

} // namespace

void mlir::triton::xpu::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfo &targetInfo,
    PatternBenefit benefit) {

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "_ZN3xpu10__fsqrt_rnEf", benefit);

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_UNARY_OP(arith::NegFOp, LLVM::FNegOp)
  POPULATE_UNARY_OP(arith::ExtFOp, LLVM::FPExtOp)
  POPULATE_UNARY_OP(arith::TruncFOp, LLVM::FPTruncOp)
  POPULATE_UNARY_OP(arith::SIToFPOp, LLVM::SIToFPOp)
  POPULATE_UNARY_OP(arith::FPToSIOp, LLVM::FPToSIOp)
  POPULATE_UNARY_OP(math::ExpOp, LLVM::Exp2Op)
  POPULATE_UNARY_OP(math::LogOp, LLVM::Log2Op)
#undef POPULATE_UNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_BINARY_OP(arith::AddFOp, LLVM::FAddOp)        // addf
  POPULATE_BINARY_OP(arith::SubFOp, LLVM::FSubOp)        // subf
  POPULATE_BINARY_OP(arith::MulFOp, LLVM::FMulOp)        // mulf
  POPULATE_BINARY_OP(arith::DivFOp, LLVM::FDivOp)        // divf
  POPULATE_BINARY_OP(arith::MaximumFOp, LLVM::MaximumOp) // maximum
  POPULATE_BINARY_OP(arith::MinimumFOp, LLVM::MinimumOp) // minimum
  POPULATE_BINARY_OP(triton::PreciseDivFOp, LLVM::FDivOp)
#undef POPULATE_BINARY_OP

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}
