#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <algorithm>
#include <cctype>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

static Type getConvertedElementType(Type type,
                                    const TypeConverter *typeConverter) {
  Type convertedType = typeConverter->convertType(type);
  if (auto vecType = dyn_cast<VectorType>(convertedType))
    return vecType.getElementType();
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(convertedType)) {
    if (!structType.getBody().empty())
      return structType.getBody().front();
  }
  return convertedType;
}

static Value maybeBitcastSameWidth(TritonLLVMOpBuilder &b, Value value,
                                   Type targetTy) {
  if (value.getType() == targetTy)
    return value;
  Type srcTy = value.getType();
  if (srcTy.isIntOrFloat() && targetTy.isIntOrFloat() &&
      srcTy.getIntOrFloatBitWidth() == targetTy.getIntOrFloatBitWidth())
    return b.bitcast(value, targetTy);
  return value;
}

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
    TritonLLVMOpBuilder b(loc, rewriter);
    Type logicalElemTy =
        this->typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy =
        getConvertedElementType(op.getType(), this->typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);

    auto intTy = dyn_cast<IntegerType>(logicalElemTy);
    auto elemWidth = intTy.getWidth();
    Value out;
    if (!intTy || (elemWidth != 16 && elemWidth != 32)) {
      out = LLVM::SDivOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
      return {maybeBitcastSameWidth(b, out, packedElemTy)};
    }

    Type widenedTy;
    if (elemWidth == 16) {
      widenedTy = rewriter.getI32Type();
      Value lhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, lhs);
      Value rhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, rhs);
      Value outWide =
          createI32SignedDivCall(op, loc, rewriter, lhsWide, rhsWide);
      out = LLVM::TruncOp::create(rewriter, loc, logicalElemTy, outWide);
    } else {
      out = createI32SignedDivCall(op, loc, rewriter, lhs, rhs);
    }
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
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
    TritonLLVMOpBuilder b(loc, rewriter);
    Type logicalElemTy =
        this->typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy =
        getConvertedElementType(op.getType(), this->typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);

    auto intTy = dyn_cast<IntegerType>(logicalElemTy);
    auto elemWidth = intTy.getWidth();
    Value out;
    if (!intTy || (elemWidth != 16 && elemWidth != 32)) {
      out = LLVM::SRemOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
      return {maybeBitcastSameWidth(b, out, packedElemTy)};
    }

    if (elemWidth == 16) {
      Type widenedTy = rewriter.getI32Type();
      Value lhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, lhs);
      Value rhsWide = LLVM::SExtOp::create(rewriter, loc, widenedTy, rhs);
      Value remWide =
          LLVM::SRemOp::create(rewriter, loc, widenedTy, lhsWide, rhsWide);
      out = LLVM::TruncOp::create(rewriter, loc, logicalElemTy, remWide);
    } else {
      Value quot = createI32SignedDivCall(op, loc, rewriter, lhs, rhs);
      Value prod = LLVM::MulOp::create(rewriter, loc, logicalElemTy, quot, rhs);
      out = LLVM::SubOp::create(rewriter, loc, logicalElemTy, lhs, prod);
    }

    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit = patternBenefitDefault)
      : Base(typeConverter, axisAnalysisPass, benefit) {}

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

  static bool isFp8Burst2Enabled() {
    std::string envValue =
        mlir::triton::tools::getStrEnv("TRITON_MUSA_ENABLE_FP8_BURST2");
    if (envValue.empty())
      return false;
    std::transform(envValue.begin(), envValue.end(), envValue.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return envValue == "1" || envValue == "true" || envValue == "on";
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

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    SmallVector<Value> ops = {v};
    SmallVector<Type, 1> resultTypes{f32_ty};
    auto intrinsic = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.musa.bfloat162float", resultTypes, ops);
    return intrinsic.getResult(0);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, RoundingMode rounding) {
    if (rounding == RoundingMode::RTZ) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto asInt32 = b.bitcast(v, i32_ty);
      auto shifted = b.lshr(i32_ty, asInt32, b.i32_val(16));
      auto truncated = b.trunc(i16_ty, shifted);
      return b.bitcast(truncated, bf16_ty);
    }
    SmallVector<Value> ops = {v};
    SmallVector<Type, 1> resultTypes{bf16_ty};
    auto intrinsic = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.musa.float2bfloat16", resultTypes, ops);
    return intrinsic.getResult(0);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, RoundingMode rounding) {
    switch (rounding) {
    case RoundingMode::RTNE:
      return LLVM::FPTruncOp::create(rewriter, loc, f16_ty, v);
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
    Type packedDstTy = getConvertedElementType(op.getType(), typeConverter);
    Value src = operands[0][0];
    if (isa<FloatType>(srcElemTy)) {
      Type logicalSrcTy = typeConverter->convertType(srcElemTy);
      src = maybeBitcastSameWidth(b, src, logicalSrcTy);
    }

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
          Type logicalSrcTy = typeConverter->convertType(srcElemTy);
          SmallVector<Value> outVals;
          outVals.reserve(operands.size());
          for (unsigned i = 0; i < operands.size(); ++i) {
            Value inVal = operands[i][0];
            if (isa<FloatType>(srcElemTy))
              inVal = maybeBitcastSameWidth(b, inVal, logicalSrcTy);
            outVals.push_back(convertFp8DowncastRTZ(loc, rewriter, op, inVal,
                                                    srcElemTy, dstElemTy));
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
          srcElemTy, dstElemTy, roundingMode, isFp8Burst2Enabled());
      Type logicalSrcTy = typeConverter->convertType(srcElemTy);
      SmallVector<Value> inVals;
      for (unsigned i = 0; i < std::min(numElements, operands.size()); ++i) {
        Value inVal = operands[i][0];
        if (isa<FloatType>(srcElemTy))
          inVal = maybeBitcastSameWidth(b, inVal, logicalSrcTy);
        inVals.push_back(inVal);
      }
      inVals.resize(numElements,
                    b.undef(typeConverter->convertType(srcElemTy)));
      auto outVals = convertFp8(getTypeConverter(), op, loc, rewriter, inVals,
                                srcElemTy, dstElemTy, funcName);
      outVals.resize(std::min(numElements, operands.size()));
      return outVals;
    }

    if (srcElemTy.isBF16() && dstElemTy.isF32()) {
      return {maybeBitcastSameWidth(b, convertBf16ToFp32(loc, rewriter, src),
                                    packedDstTy)};
    }
    if (srcElemTy.isF32() && dstElemTy.isBF16()) {
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      return {maybeBitcastSameWidth(
          b, convertFp32ToBf16(loc, rewriter, src, rounding), packedDstTy)};
    }
    if (srcElemTy.isF16() && dstElemTy.isF32()) {
      Value out = LLVM::FPExtOp::create(rewriter, loc, f32_ty, src);
      return {maybeBitcastSameWidth(b, out, packedDstTy)};
    }
    if (srcElemTy.isF32() && dstElemTy.isF16()) {
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      Value out = convertFp32ToFp16(loc, rewriter, src, rounding);
      return {maybeBitcastSameWidth(b, out, packedDstTy)};
    }
    if (srcElemTy.isF16() && dstElemTy.isBF16()) {
      Value tmp = LLVM::FPExtOp::create(rewriter, loc, f32_ty, src);
      auto rounding = op.getRounding().value_or(RoundingMode::RTNE);
      Value out = convertFp32ToBf16(loc, rewriter, tmp, rounding);
      return {maybeBitcastSameWidth(b, out, packedDstTy)};
    }
    if (srcElemTy.isBF16() && dstElemTy.isF16()) {
      Value tmp = convertBf16ToFp32(loc, rewriter, src);
      Value out = LLVM::FPTruncOp::create(rewriter, loc, f16_ty, tmp);
      return {maybeBitcastSameWidth(b, out, packedDstTy)};
    }
    if (srcElemTy == dstElemTy) {
      return {src};
    }

    // Fallback to LLVM FP trunc/ext when applicable.
    if (isa<FloatType>(srcElemTy) && isa<FloatType>(dstElemTy)) {
      if (srcElemTy.getIntOrFloatBitWidth() < dstElemTy.getIntOrFloatBitWidth())
        return {maybeBitcastSameWidth(
            b,
            LLVM::FPExtOp::create(rewriter, loc,
                                  typeConverter->convertType(dstElemTy), src),
            packedDstTy)};
      if (srcElemTy.getIntOrFloatBitWidth() > dstElemTy.getIntOrFloatBitWidth())
        return {maybeBitcastSameWidth(
            b,
            LLVM::FPTruncOp::create(rewriter, loc,
                                    typeConverter->convertType(dstElemTy), src),
            packedDstTy)};
    }

    return {};
  }
};

template <typename OpType>
Value emitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto lhs =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto rhs =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = OpType::create(rewriter, loc, f32_ty, lhs, rhs);
  return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, result,
                                               RoundingMode::RTNE);
}

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
    TritonLLVMOpBuilder b(loc, rewriter);
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Value input = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);

    Type f64Ty = rewriter.getF64Type();
    Type f32Ty = rewriter.getF32Type();
    Value inputF64 = LLVM::FPExtOp::create(rewriter, loc, f64Ty, input);

    Type funcType = getFunctionType(f64Ty, ValueRange{inputF64});
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, "__mt_sqrt_f64", funcType);
    Value sqrtResultF64 =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{inputF64})
            .getResult();
    Value resultF32 =
        LLVM::FPTruncOp::create(rewriter, loc, f32Ty, sqrtResultF64);
    return {maybeBitcastSameWidth(b, resultF32, packedElemTy)};
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
    TritonLLVMOpBuilder b(loc, rewriter);
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);

    // precise_divf is defined for f32; compute in f64 and cast back for
    // improved precision.
    Type f64Ty = rewriter.getF64Type();
    Type f32Ty = rewriter.getF32Type();
    Value lhsF64 = LLVM::FPExtOp::create(rewriter, loc, f64Ty, lhs);
    Value rhsF64 = LLVM::FPExtOp::create(rewriter, loc, f64Ty, rhs);
    Value divResultF64 =
        LLVM::FDivOp::create(rewriter, loc, f64Ty, lhsF64, rhsF64);
    Value resultF32 =
        LLVM::FPTruncOp::create(rewriter, loc, f32Ty, divResultF64);
    return {maybeBitcastSameWidth(b, resultF32, packedElemTy)};
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {emitDualBF16ElementwiseOp<LLVM::FDivOp>(loc, rewriter, operands)};
    }
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);
    Value out = LLVM::FDivOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {emitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
    }
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);
    Value out = LLVM::FMulOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {emitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    }
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);
    Value out = LLVM::FAddOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {emitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    }
    Type logicalElemTy =
        typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy = getConvertedElementType(op.getType(), typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);
    Value out = LLVM::FSubOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

template <typename SrcOp, typename DstOp>
struct FPBinaryBitcastOpConversion
    : ElementwiseOpConversionBase<SrcOp,
                                  FPBinaryBitcastOpConversion<SrcOp, DstOp>> {
  using Base =
      ElementwiseOpConversionBase<SrcOp,
                                  FPBinaryBitcastOpConversion<SrcOp, DstOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(SrcOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Type logicalElemTy =
        this->typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy =
        getConvertedElementType(op.getType(), this->typeConverter);
    Value lhs = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value rhs = maybeBitcastSameWidth(b, operands[0][1], logicalElemTy);
    Value out = DstOp::create(rewriter, loc, logicalElemTy, lhs, rhs);
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

template <typename SrcOp, typename DstOp>
struct FPUnaryBitcastOpConversion
    : ElementwiseOpConversionBase<SrcOp,
                                  FPUnaryBitcastOpConversion<SrcOp, DstOp>> {
  using Base =
      ElementwiseOpConversionBase<SrcOp,
                                  FPUnaryBitcastOpConversion<SrcOp, DstOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(SrcOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Type logicalElemTy =
        this->typeConverter->convertType(getElementType(op.getResult()));
    Type packedElemTy =
        getConvertedElementType(op.getType(), this->typeConverter);
    Value src = maybeBitcastSameWidth(b, operands[0][0], logicalElemTy);
    Value out = DstOp::create(rewriter, loc, logicalElemTy, src,
                              adaptor.getAttributes().getValue());
    return {maybeBitcastSameWidth(b, out, packedElemTy)};
  }
};

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      Value f32Val =
          LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, f32Val,
                                                    RoundingMode::RTNE)};
    }
    return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (inElemTy.isBF16()) {
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
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto inElemTy = getElementType(op.getIn());
    auto outElemTy = getElementType(op.getOut());
    Type packedOutTy = getConvertedElementType(op.getType(), typeConverter);
    Value src = operands[0][0];
    if (isa<FloatType>(inElemTy)) {
      Type logicalInTy = typeConverter->convertType(inElemTy);
      src = maybeBitcastSameWidth(b, src, logicalInTy);
    }

    Value out;
    if (inElemTy.isBF16()) {
      out = FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, src);
      return {maybeBitcastSameWidth(b, out, packedOutTy)};
    }
    Type logicalOutTy = typeConverter->convertType(outElemTy);
    out = LLVM::FPExtOp::create(rewriter, loc, logicalOutTy, src);
    return {maybeBitcastSameWidth(b, out, packedOutTy)};
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto inElemTy = getElementType(op.getIn());
    auto outElemTy = getElementType(op.getOut());
    Type packedOutTy = getConvertedElementType(op.getType(), typeConverter);
    Value src = operands[0][0];
    if (isa<FloatType>(inElemTy)) {
      Type logicalInTy = typeConverter->convertType(inElemTy);
      src = maybeBitcastSameWidth(b, src, logicalInTy);
    }

    Value out;
    if (outElemTy.isBF16()) {
      out = FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, src,
                                                  RoundingMode::RTNE);
      return {maybeBitcastSameWidth(b, out, packedOutTy)};
    }
    Type logicalOutTy = typeConverter->convertType(outElemTy);
    out = LLVM::FPTruncOp::create(rewriter, loc, logicalOutTy, src);
    return {maybeBitcastSameWidth(b, out, packedOutTy)};
  }
};

} // namespace

void mlir::triton::MUSA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int /*computeCapability*/,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit priorityBenefit(benefit.getBenefit() + 1);
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis,
                                 priorityBenefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis,
                                 priorityBenefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis,
                                 priorityBenefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis,
                                 priorityBenefit);
  patterns.add<DivSIOpConversion>(typeConverter, axisInfoAnalysis,
                                  priorityBenefit);
  patterns.add<RemSIOpConversion>(typeConverter, axisInfoAnalysis,
                                  priorityBenefit);
  patterns.add<FPBinaryBitcastOpConversion<arith::MaxNumFOp, LLVM::MaxNumOp>>(
      typeConverter, axisInfoAnalysis, priorityBenefit);
  patterns.add<FPBinaryBitcastOpConversion<arith::MinNumFOp, LLVM::MinNumOp>>(
      typeConverter, axisInfoAnalysis, priorityBenefit);
  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<PreciseSqrtOpConversion>(typeConverter, axisInfoAnalysis,
                                        priorityBenefit);
  patterns.add<PreciseDivOpConversion>(typeConverter, axisInfoAnalysis,
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
