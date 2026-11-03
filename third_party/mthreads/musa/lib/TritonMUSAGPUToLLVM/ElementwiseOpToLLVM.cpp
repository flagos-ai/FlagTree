#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

static Value createI32SignedDivCall(Operation *op, Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    Value lhs, Value rhs) {
  Type funcType = LLVM::LLVMFunctionType::get(i32_ty, {i32_ty, i32_ty});
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, op, "__mt_sdiv_i32", funcType);
  return LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{lhs, rhs})
      .getResult();
}

struct DivSIOpConversion
    : ElementwiseOpConversionBase<arith::DivSIOp, DivSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivSIOp, DivSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivSIOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Value lhs = operands[0][0];
    Value rhs = operands[0][1];

    auto intTy = dyn_cast<IntegerType>(elemTy);
    if (!intTy || intTy.getWidth() != 16) {
      SmallVector<Value> divOperands{lhs, rhs};
      Value out = LLVM::SDivOp::create(rewriter, loc, elemTy, divOperands,
                                       op->getAttrs());
      return {out};
    }

    Type widenedTy = rewriter.getI32Type();
    Value lhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, lhs);
    Value rhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, rhs);
    Value outWide = createI32SignedDivCall(op, loc, rewriter, lhsWide, rhsWide);
    Value out = LLVM::TruncOp::create(rewriter, loc, elemTy, outWide);
    return {out};
  }
};

struct RemSIOpConversion
    : ElementwiseOpConversionBase<arith::RemSIOp, RemSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::RemSIOp, RemSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::RemSIOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Value lhs = operands[0][0];
    Value rhs = operands[0][1];

    auto intTy = dyn_cast<IntegerType>(elemTy);
    if (!intTy || intTy.getWidth() != 16) {
      Value out = LLVM::SRemOp::create(rewriter, loc, elemTy, lhs, rhs);
      return {out};
    }

    Type widenedTy = rewriter.getI32Type();
    Value lhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, lhs);
    Value rhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, rhs);
    Value remWide =
        LLVM::SRemOp::create(rewriter, loc, widenedTy, lhsWide, rhsWide);
    Value out = LLVM::TruncOp::create(rewriter, loc, elemTy, remWide);
    return {out};
  }
};

struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion>;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              bool enableFp8Burst2,
                              PatternBenefit benefit = patternBenefitDefault,
                              bool nativeBf16Cvt = false)
      : Base(typeConverter, axisAnalysisPass, benefit),
        nativeBf16Cvt(nativeBf16Cvt), enableFp8Burst2(enableFp8Burst2) {}

  const bool nativeBf16Cvt;

  struct Fp8ConversionDesc {
    StringRef funcName;
    size_t numElements;
  };

  static bool isFp8Type(Type ty) {
    return isa<Float8E4M3FNType, Float8E5M2Type>(ty);
  }

  static StringRef getFp8ConversionIntrinsic(StringRef funcName) {
    if (funcName == "__mt_tt_f16_to_e4m3")
      return "llvm.musa.f162e4m3.rn";
    if (funcName == "__mt_tt_f16_to_e5m2")
      return "llvm.musa.f162e5m2.rn";
    return {};
  }

  std::pair<StringRef, size_t>
  getFp8ConversionFunc(Type srcTy, Type dstTy,
                       std::optional<RoundingMode> roundingMode,
                       bool enableFp8Burst2) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>,
                    SmallVector<Fp8ConversionDesc>>
        conversionTable = {
            // F8 -> F32
            {{F8E4M3TyID, F32TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2f32", 2}, {"__mt_tt_e4m3_to_f32", 1}}},
            {{F8E5M2TyID, F32TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2f32", 2}, {"__mt_tt_e5m2_to_f32", 1}}},
            // F8 -> F16
            {{F8E4M3TyID, F16TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2f16", 2}, {"__mt_tt_e4m3_to_f16", 1}}},
            {{F8E5M2TyID, F16TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2f16", 2}, {"__mt_tt_e5m2_to_f16", 1}}},
            // F8 -> BF16
            {{F8E4M3TyID, BF16TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2bf16", 2}, {"__mt_tt_e4m3_to_bf16", 1}}},
            {{F8E5M2TyID, BF16TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2bf16", 2}, {"__mt_tt_e5m2_to_bf16", 1}}},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f32_to_v2e4m3", 2}, {"__mt_tt_f32_to_e4m3", 1}}},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f32_to_v2e5m2", 2}, {"__mt_tt_f32_to_e5m2", 1}}},
            // F16 -> F8
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f16_to_v2e4m3", 2}, {"__mt_tt_f16_to_e4m3", 1}}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f16_to_v2e5m2", 2}, {"__mt_tt_f16_to_e5m2", 1}}},
            // BF16 -> F8
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2bf16_to_v2e4m3", 2}, {"__mt_tt_bf16_to_e4m3", 1}}},
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2bf16_to_v2e5m2", 2}, {"__mt_tt_bf16_to_e5m2", 1}}},
        };

    auto key = std::make_tuple(srcTy.getTypeID(), dstTy.getTypeID(),
                               roundingMode.value_or(undefRounding));
    auto it = conversionTable.find(key);
    if (it == conversionTable.end()) {
      llvm::report_fatal_error("Unsupported MUSA fp8 conversion");
    }
    const auto &entries = it->second;
    const auto &entry = enableFp8Burst2 ? entries.front() : entries.back();
    return {entry.funcName, entry.numElements};
  }

  static SmallVector<Value>
  convertFp8(const LLVMTypeConverter *typeConverter, FpToFpOp op, Location loc,
             ConversionPatternRewriter &rewriter, const SmallVector<Value> &v,
             Type srcElementType, Type dstElementType, StringRef funcName) {
    TritonLLVMOpBuilder b(loc, rewriter);
    const size_t numElements = v.size();
    Type inpType;
    Type outType;
    Value inVals;

    if (numElements == 1) {
      inpType = typeConverter->convertType(srcElementType);
      outType = typeConverter->convertType(dstElementType);
      inVals = v[0];
    } else {
      inpType = vec_ty(typeConverter->convertType(srcElementType), numElements);
      outType = vec_ty(typeConverter->convertType(dstElementType), numElements);
      inVals = b.undef(inpType);
      for (size_t i = 0; i < numElements; ++i)
        inVals = b.insert_element(inpType, inVals, v[i], b.i32_val(i));
    }

    Value outVals;
    if (numElements == 1) {
      if (StringRef intrinsicName = getFp8ConversionIntrinsic(funcName);
          !intrinsicName.empty()) {
        auto intrinsic = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsicName, TypeRange{outType},
            ValueRange{inVals});
        outVals = intrinsic.getResult(0);
      }
    }
    if (!outVals) {
      Type funcType = LLVM::LLVMFunctionType::get(outType, inpType);
      LLVM::LLVMFuncOp funcOp =
          appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
      outVals =
          LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{inVals})
              .getResult();
    }

    SmallVector<Value> ret;
    Type outElemType = typeConverter->convertType(dstElementType);
    for (size_t i = 0; i < numElements; ++i) {
      ret.push_back(numElements == 1 ? outVals
                                     : b.extract_element(outElemType, outVals,
                                                         b.i32_val(i)));
    }
    return ret;
  }

  // The intrinsic selects the same FOP.MOV as a bare fpext/fptrunc on PH1+,
  // and the call boundary stops InstCombine from fusing conversions into
  // forms mtcc cannot select (e.g. fpext bfloat->double).
  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return LLVM::createLLVMIntrinsicCallOp(
               rewriter, loc, "llvm.musa.bfloat162float", f32_ty, {v})
        .getResult(0);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, RoundingMode rounding,
                                 bool nativeBf16) {
    if (rounding == RoundingMode::RTZ) {
      if (nativeBf16)
        return LLVM::createLLVMIntrinsicCallOp(
                   rewriter, loc, "llvm.musa.float2bfloat16.rz", bf16_ty, {v})
            .getResult(0);
      TritonLLVMOpBuilder b(loc, rewriter);
      auto asInt32 = b.bitcast(v, i32_ty);
      auto shifted = b.lshr(i32_ty, asInt32, b.i32_val(16));
      auto truncated = b.trunc(i16_ty, shifted);
      return b.bitcast(truncated, bf16_ty);
    }
    return LLVM::createLLVMIntrinsicCallOp(
               rewriter, loc, "llvm.musa.float2bfloat16", bf16_ty, {v})
        .getResult(0);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, RoundingMode rounding) {
    switch (rounding) {
    case RoundingMode::RTNE:
      return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.f2h.rn",
                                             f16_ty, {v})
          .getResult(0);
    case RoundingMode::RTZ:
      return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.f2h.rz",
                                             f16_ty, {v})
          .getResult(0);
    default:
      emitError(loc) << "unsupported rounding mode for f32->f16 conversion: "
                     << stringifyRoundingMode(rounding);
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->f16 conversion");
    }
  }

  static Value convertSrcFpToFp32(Location loc,
                                  ConversionPatternRewriter &rewriter, Value v,
                                  Type srcTy) {
    if (srcTy.isF32())
      return v;
    if (srcTy.isF16())
      return LLVM::FPExtOp::create(rewriter, loc, f32_ty, v);
    if (srcTy.isBF16())
      return convertBf16ToFp32(loc, rewriter, v);
    llvm::report_fatal_error("Unsupported MUSA fp8 RTZ source type");
  }

  static Value clearFp32SignBit(Location loc,
                                ConversionPatternRewriter &rewriter, Value v) {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value bits = b.bitcast(v, i32_ty);
    Value absBits = b.and_(bits, b.i32_val(0x7fffffff));
    return b.bitcast(absBits, f32_ty);
  }

  static Value convertFp8Scalar(const LLVMTypeConverter *typeConverter,
                                FpToFpOp op, Location loc,
                                ConversionPatternRewriter &rewriter, Value v,
                                Type srcElementType, Type dstElementType,
                                StringRef funcName) {
    auto outVals = convertFp8(typeConverter, op, loc, rewriter, {v},
                              srcElementType, dstElementType, funcName);
    assert(outVals.size() == 1 && "expected scalar fp8 conversion");
    return outVals.front();
  }

  Value convertFp8DowncastRTZ(Location loc, ConversionPatternRewriter &rewriter,
                              FpToFpOp op, Value src, Type srcElemTy,
                              Type dstElemTy) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value srcFp32 = convertSrcFpToFp32(loc, rewriter, src, srcElemTy);

    auto [downcastFuncName, _] =
        getFp8ConversionFunc(f32_ty, dstElemTy, RoundingMode::RTNE, false);
    Value rtneFp8 =
        convertFp8Scalar(getTypeConverter(), op, loc, rewriter, srcFp32, f32_ty,
                         dstElemTy, downcastFuncName);

    auto [upcastFuncName, __] =
        getFp8ConversionFunc(dstElemTy, f32_ty, std::nullopt, false);
    Value rtneRoundTrip =
        convertFp8Scalar(getTypeConverter(), op, loc, rewriter, rtneFp8,
                         dstElemTy, f32_ty, upcastFuncName);

    Value absSrc = clearFp32SignBit(loc, rewriter, srcFp32);
    Value absRtneRoundTrip = clearFp32SignBit(loc, rewriter, rtneRoundTrip);
    Value roundedAwayFromZero = b.fcmp_ogt(absRtneRoundTrip, absSrc);
    Value rtzFp8 = b.sub(rtneFp8, b.i8_val(1));
    return b.select(roundedAwayFromZero, rtzFp8, rtneFp8);
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto srcElemTy = getElementTypeOrSelf(op.getSrc().getType());
    auto dstElemTy = getElementTypeOrSelf(op.getType());
    Value src = operands[0][0];

    auto roundingMode = op.getRounding();

    bool isFp8Conversion = isFp8Type(srcElemTy) || isFp8Type(dstElemTy);
    if (isFp8Conversion) {
      if (isFp8Type(dstElemTy)) {
        if (roundingMode.has_value() &&
            roundingMode.value() == RoundingMode::RTZ) {
          if (!isa<Float16Type, BFloat16Type, Float32Type>(srcElemTy)) {
            llvm::report_fatal_error(
                "Unsupported MUSA fp8 RTZ downcast source type");
          }
          SmallVector<Value> outVals;
          outVals.reserve(operands.size());
          for (unsigned i = 0; i < operands.size(); ++i) {
            outVals.push_back(convertFp8DowncastRTZ(
                loc, rewriter, op, operands[i][0], srcElemTy, dstElemTy));
          }
          return outVals;
        }
        if (!roundingMode.has_value() ||
            roundingMode.value() != RoundingMode::RTNE) {
          llvm::report_fatal_error(
              "MUSA fp8 downcast requires RTNE rounding mode");
        }
      }
      auto [funcName, numElements] = getFp8ConversionFunc(
          srcElemTy, dstElemTy, roundingMode, enableFp8Burst2);
      SmallVector<Value> inVals;
      for (unsigned i = 0; i < std::min(numElements, operands.size()); ++i)
        inVals.push_back(operands[i][0]);
      inVals.resize(numElements,
                    b.undef(typeConverter->convertType(srcElemTy)));
      auto outVals = convertFp8(getTypeConverter(), op, loc, rewriter, inVals,
                                srcElemTy, dstElemTy, funcName);
      outVals.resize(std::min(numElements, operands.size()));
      return outVals;
    }

    if (srcElemTy.isBF16() && dstElemTy.isF32()) {
      return {convertBf16ToFp32(loc, rewriter, src)};
    }
    if (srcElemTy.isF32() && dstElemTy.isBF16()) {
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      return {convertFp32ToBf16(loc, rewriter, src, rounding, nativeBf16Cvt)};
    }
    if (srcElemTy.isF16() && dstElemTy.isF32()) {
      return {LLVM::FPExtOp::create(rewriter, loc, f32_ty, src)};
    }
    if (srcElemTy.isF32() && dstElemTy.isF16()) {
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      return {convertFp32ToFp16(loc, rewriter, src, rounding)};
    }
    if (srcElemTy.isF16() && dstElemTy.isBF16()) {
      Value tmp = LLVM::FPExtOp::create(rewriter, loc, f32_ty, src);
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      return {convertFp32ToBf16(loc, rewriter, tmp, rounding, nativeBf16Cvt)};
    }
    if (srcElemTy.isBF16() && dstElemTy.isF16()) {
      Value tmp = convertBf16ToFp32(loc, rewriter, src);
      return {convertFp32ToFp16(loc, rewriter, tmp, RoundingMode::RTNE)};
    }
    if (srcElemTy == dstElemTy) {
      return {src};
    }

    // Fallback to LLVM FP trunc/ext when applicable.
    if (isa<FloatType>(srcElemTy) && isa<FloatType>(dstElemTy)) {
      if (srcElemTy.getIntOrFloatBitWidth() < dstElemTy.getIntOrFloatBitWidth())
        return {LLVM::FPExtOp::create(
            rewriter, loc, typeConverter->convertType(dstElemTy), src)};
      if (srcElemTy.getIntOrFloatBitWidth() > dstElemTy.getIntOrFloatBitWidth())
        return {LLVM::FPTruncOp::create(
            rewriter, loc, typeConverter->convertType(dstElemTy), src)};
    }

    return {};
  }

private:
  bool enableFp8Burst2 = false;
};

// Lower triton.precise_sqrt through f64 for better f32 precision.
struct PreciseSqrtOpConversion
    : public ElementwiseOpConversionBase<triton::PreciseSqrtOp,
                                         PreciseSqrtOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::PreciseSqrtOp,
                                           PreciseSqrtOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit PreciseSqrtOpConversion(LLVMTypeConverter &typeConverter,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : Base(typeConverter, axisAnalysisPass, benefit) {}

  SmallVector<Value> createDestOps(triton::PreciseSqrtOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // precise_sqrt is defined for f32; compute in f64 and cast back for
    // improved precision.
    Type f64Ty = rewriter.getF64Type();
    Type f32Ty = rewriter.getF32Type();
    Value inputF64 =
        LLVM::FPExtOp::create(rewriter, loc, f64Ty, operands[0][0]);

    Type funcType = getFunctionType(f64Ty, ValueRange{inputF64});
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, "__mt_sqrt_f64", funcType);
    Value sqrtResultF64 =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{inputF64})
            .getResult();
    Value resultF32 =
        LLVM::FPTruncOp::create(rewriter, loc, f32Ty, sqrtResultF64);
    return {resultF32};
  }
};

struct PreciseDivOpConversion
    : public ElementwiseOpConversionBase<triton::PreciseDivFOp,
                                         PreciseDivOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::PreciseDivFOp,
                                           PreciseDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit PreciseDivOpConversion(LLVMTypeConverter &typeConverter,
                                  ModuleAxisInfoAnalysis &axisAnalysisPass,
                                  PatternBenefit benefit)
      : Base(typeConverter, axisAnalysisPass, benefit) {}

  SmallVector<Value> createDestOps(triton::PreciseDivFOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // precise_divf is defined for f32; compute in f64 and cast back for
    // improved precision.
    Type f64Ty = rewriter.getF64Type();
    Type f32Ty = rewriter.getF32Type();
    Value lhsF64 = LLVM::FPExtOp::create(rewriter, loc, f64Ty, operands[0][0]);
    Value rhsF64 = LLVM::FPExtOp::create(rewriter, loc, f64Ty, operands[0][1]);
    Value divResultF64 =
        LLVM::FDivOp::create(rewriter, loc, f64Ty, lhsF64, rhsF64);
    Value resultF32 =
        LLVM::FPTruncOp::create(rewriter, loc, f32Ty, divResultF64);
    return {resultF32};
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit FDivOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit, bool nativeBf16Cvt)
      : Base(typeConverter, axisAnalysisPass, benefit),
        nativeBf16Cvt(nativeBf16Cvt) {}

  const bool nativeBf16Cvt;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Only tl.fdiv reaches here with bf16 (`/` is promoted to f32 by the
    // frontend). mtcc cannot expand fdiv bfloat on QY1/QY2, and PH1's
    // expansion is identical to this promote, so keep it for all targets.
    if (getElementType(op.getLhs()).isBF16() &&
        getElementType(op.getRhs()).isBF16()) {
      auto lhs =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      auto rhs =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][1]);
      auto result = LLVM::FDivOp::create(rewriter, loc, f32_ty, lhs, rhs);
      return {FpToFpOpConversion::convertFp32ToBf16(
          loc, rewriter, result, RoundingMode::RTNE, nativeBf16Cvt)};
    }
    Value out = LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                     operands[0][1]);
    return {out};
  }
};

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit SIToFPOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit, bool nativeBf16Cvt)
      : Base(typeConverter, axisAnalysisPass, benefit),
        nativeBf16Cvt(nativeBf16Cvt) {}

  const bool nativeBf16Cvt;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Pre-PH1 cannot select a bare bf16 cvt; detour through f32 explicitly.
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && !nativeBf16Cvt) {
      Value f32Val =
          LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(
          loc, rewriter, f32Val, RoundingMode::RTNE, nativeBf16Cvt)};
    }
    return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit FPToSIOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit, bool nativeBf16Cvt)
      : Base(typeConverter, axisAnalysisPass, benefit),
        nativeBf16Cvt(nativeBf16Cvt) {}

  const bool nativeBf16Cvt;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Pre-PH1 detours through f32 like SIToFP. mtcc currently mis-lowers
    // fptosi bfloat -> 64-bit int (test_int_bf16_cast.py xfail).
    Type inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16() && !nativeBf16Cvt) {
      Value f32Val =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, f32Val)};
    }
    return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  using Base::Base;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    Value src = operands[0][0];
    if (inElemTy.isBF16()) {
      // The frontend splits wider bf16 conversions at f32.
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, src)};
    }
    Value out = LLVM::FPExtOp::create(rewriter, loc, elemTy, src);
    return {out};
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit TruncFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit, bool nativeBf16Cvt)
      : Base(typeConverter, axisAnalysisPass, benefit),
        nativeBf16Cvt(nativeBf16Cvt) {}

  const bool nativeBf16Cvt;

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    auto outElemTy = getElementType(op.getOut());
    Value src = operands[0][0];
    // The bf16 helper takes an f32 input; wider sources fall through to a
    // bare fptrunc for mtcc to legalize.
    if (outElemTy.isBF16() && inElemTy.isF32())
      return {FpToFpOpConversion::convertFp32ToBf16(
          loc, rewriter, src, RoundingMode::RTNE, nativeBf16Cvt)};
    Value out = LLVM::FPTruncOp::create(rewriter, loc, elemTy, src);
    return {out};
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (getIntOrFloatOrPtrBitWidth(elemTy) != 32)
      return {};

    TritonLLVMOpBuilder b(loc, rewriter);
    constexpr double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));
    return {math::Exp2Op::create(rewriter, loc, f32_ty, prod,
                                 adaptor.getAttributes().getValue())};
  }
};

} // namespace

void mlir::triton::MUSA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    bool enableFp8Burst2, const TargetInfo &targetInfo,
    PatternBenefit benefit) {
  PatternBenefit priorityBenefit(benefit.getBenefit() + 1);
  // Pre-PH1 cannot select plain bf16 cvt ops and uses intrinsics instead.
  const bool nativeBf16Cvt = computeCapability >= 31;
  // Use higher priority than generic TritonGPU elementwise lowering so the
  // MUSA-specific FP/bfloat/math lowering is selected first.
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis,
                                 priorityBenefit, nativeBf16Cvt);
  patterns.add<DivSIOpConversion>(typeConverter, axisInfoAnalysis,
                                  priorityBenefit);
  patterns.add<RemSIOpConversion>(typeConverter, axisInfoAnalysis,
                                  priorityBenefit);
#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, priorityBenefit)
  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);
#undef POPULATE_OP
  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit,
                                   nativeBf16Cvt);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit,
                                   nativeBf16Cvt);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit,
                                   nativeBf16Cvt);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   enableFp8Burst2, benefit, nativeBf16Cvt);
  patterns.add<PreciseSqrtOpConversion>(typeConverter, axisInfoAnalysis,
                                        priorityBenefit);
  patterns.add<PreciseDivOpConversion>(typeConverter, axisInfoAnalysis,
                                       priorityBenefit);
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis,
                                      priorityBenefit);
  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  bool hwNanPropagationSupported = targetInfo.supportMaximumMinimum();
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}
