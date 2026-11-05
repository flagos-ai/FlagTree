#include "PatternPopulation.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace Intellifusion {

// lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp.
struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
  using ConvertOpToLLVMPattern<AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = dyn_cast<RankedTensorType>(resultTy);
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(
          cast<PointerType>(resultTensorTy.getElementType()).getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = b.gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(isa<PointerType>(resultTy));
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          cast<PointerType>(resultTy).getPointeeType());
      Value result = b.gep(resultPtrTy, resultElemTy, adaptor.getPtr(),
                           adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

// Reject tl.fdiv(float16) with a conversion diagnostic.
struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    unsigned bitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
    StringRef name;
    Type resultTy;
    if (bitwidth == 32) {
      name = "llvm.nvvm.div.full";
      resultTy = f32_ty;
    } else if (bitwidth == 64) {
      name = "llvm.nvvm.div.rn.d";
      resultTy = f64_ty;
    } else if (elemTy.isF16()) {
      op.emitError("unsupported f16 arith.divf lowering for GPNPU");
      return {};
    } else {
      return {LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }

    Value args[] = {operands[0][0], operands[0][1]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
  }
};

// PropagateNan::NONE selects the positive symmetric bound for NaN
// inputs because IFWA arithmetic canonicalizes intermediate NaNs as negative.
struct SymmetricClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, SymmetricClampFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<ClampFOp, SymmetricClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static std::optional<double> getSplatInitializer(Value value) {
    DenseTypedElementsAttr denseAttr;
    if (matchPattern(value, m_Constant(&denseAttr))) {
      if (denseAttr.isSplat())
        return denseAttr.getSplatValue<APFloat>().convertToDouble();
      return std::nullopt;
    }
    FloatAttr floatAttr;
    if (matchPattern(value, m_Constant(&floatAttr)))
      return floatAttr.getValue().convertToDouble();
    return std::nullopt;
  }

  static bool isSymmetricClamp(ClampFOp op) {
    if (auto negOp = op.getOperand(1).getDefiningOp<arith::NegFOp>()) {
      if (negOp.getOperand() == op.getOperand(2))
        return true;
    }

    auto lowerSplat = op.getOperand(1).getDefiningOp<SplatOp>();
    auto upperSplat = op.getOperand(2).getDefiningOp<SplatOp>();
    if (lowerSplat && upperSplat) {
      auto negOp = lowerSplat.getSrc().getDefiningOp<arith::NegFOp>();
      if (negOp && negOp.getOperand() == upperSplat.getSrc())
        return true;
    }

    if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
      if (subOp.getOperand(1) == op.getOperand(2)) {
        auto zero = getSplatInitializer(subOp.getOperand(0));
        if (zero && *zero == 0.0)
          return true;
      }
    }

    auto lower = getSplatInitializer(op.getOperand(1));
    auto upper = getSplatInitializer(op.getOperand(2));
    return lower && upper && *lower == -*upper;
  }

  SmallVector<Value> createDestOps(ClampFOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (!isSymmetricClamp(op))
      return {};

    bool propagateNan = op.getPropagateNan() == PropagateNan::ALL;
    llvm::Intrinsic::ID intrinsic;
    if (elemTy.isF32()) {
      intrinsic = propagateNan ? llvm::Intrinsic::nvvm_fmin_nan_xorsign_abs_f
                               : llvm::Intrinsic::nvvm_fmin_xorsign_abs_f;
    } else if (elemTy.isF16()) {
      intrinsic = propagateNan ? llvm::Intrinsic::nvvm_fmin_nan_xorsign_abs_f16
                               : llvm::Intrinsic::nvvm_fmin_xorsign_abs_f16;
    } else {
      return {};
    }

    Value input = operands[0][0];
    Value upper = operands[0][2];
    auto call = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, llvm::Intrinsic::getName(intrinsic), elemTy,
        ValueRange{input, upper});
    Value result = call.getResult(0);
    if (op.getPropagateNan() == PropagateNan::NONE) {
      Value isNan = LLVM::FCmpOp::create(
          rewriter, loc, LLVM::FCmpPredicate::une, input, input);
      result = LLVM::SelectOp::create(rewriter, loc, isNan, upper, result);
    }
    return {result};
  }
};

// Use NVVM packed conversion ops where the dialect can represent the
// conversion, and inline PTX for the scalar F16 -> BF16 conversion that NVVM
// does not expose. The IFWA toolchain lowers both forms to native conversion
// instructions. FP8 lanes are packed low byte first.
struct FpToFpOpConversion
    : ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using Base = ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static bool isFp8Type(Type type) {
    return isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
               Float8E5M2FNUZType>(type);
  }

  static SmallVector<Value> getConversionPair(MultipleOperandsRange operands) {
    SmallVector<Value> values;
    size_t pairSize = std::min<size_t>(2, operands.size());
    values.reserve(pairSize);
    for (size_t i = 0; i < pairSize; ++i)
      values.push_back(operands[i][0]);
    return values;
  }

  LogicalResult
  matchAndRewrite(FpToFpOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcTy = getElementTypeOrSelf(op.getSrc());
    Type dstTy = getElementTypeOrSelf(op.getResult());
    auto rounding = op.getRounding();
    bool involvesFp8 = isFp8Type(srcTy) || isFp8Type(dstTy);
    bool supportedFp8Conversion =
        (isa<Float8E4M3FNType, Float8E5M2Type>(srcTy) &&
         (dstTy.isF16() || dstTy.isBF16() || dstTy.isF32()) &&
         !rounding.has_value()) ||
        ((srcTy.isF16() || srcTy.isBF16() || srcTy.isF32()) &&
         isa<Float8E4M3FNType, Float8E5M2Type>(dstTy) &&
         rounding == RoundingMode::RTNE) ||
        (srcTy.isF32() && isa<Float8E5M2Type>(dstTy) &&
         rounding == RoundingMode::RTZ);
    bool supportedStandardConversion =
        (isa<Float16Type, BFloat16Type>(srcTy) && dstTy.isF32() &&
         !rounding.has_value()) ||
        (srcTy.isF32() && isa<Float16Type, BFloat16Type>(dstTy) &&
         (rounding == RoundingMode::RTNE || rounding == RoundingMode::RTZ));
    bool supportedConversion =
        involvesFp8 ? supportedFp8Conversion : supportedStandardConversion;
    if (!supportedConversion) {
      auto diagnostic =
          op.emitOpError("unsupported GPNPU floating-point conversion: ");
      diagnostic << "source=" << srcTy << ", destination=" << dstTy
                 << ", rounding=";
      if (rounding)
        diagnostic << stringifyRoundingMode(*rounding);
      else
        diagnostic << "none";
      return failure();
    }

    return Base::matchAndRewrite(op, adaptor, rewriter);
  }

  static SmallVector<Value>
  convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                    ArrayRef<Value> values, RoundingMode rounding) {
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    Type f16x2Ty = VectorType::get({2}, f16_ty);
    NVVM::FPRoundingMode nvvmRounding = rounding == RoundingMode::RTNE
                                            ? NVVM::FPRoundingMode::RN
                                            : NVVM::FPRoundingMode::RZ;
    for (size_t i = 0; i < values.size(); i += 2) {
      Value lo = values[i];
      Value hi =
          i + 1 < values.size() ? values[i + 1] : llvmBuilder.f32_val(0.0);
      Value converted =
          NVVM::ConvertF32x2ToF16x2Op::create(
              rewriter, loc, f16x2Ty, hi, lo, Value(), nvvmRounding,
              NVVM::SaturationMode::NONE, /*relu=*/false)
              .getDst();
      result.push_back(llvmBuilder.extract_element(f16_ty, converted,
                                                   llvmBuilder.i32_val(0)));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.extract_element(f16_ty, converted,
                                                     llvmBuilder.i32_val(1)));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                    ArrayRef<Value> values, RoundingMode rounding) {
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    Type bf16x2Ty = VectorType::get({2}, bf16_ty);
    NVVM::FPRoundingMode nvvmRounding = rounding == RoundingMode::RTNE
                                            ? NVVM::FPRoundingMode::RN
                                            : NVVM::FPRoundingMode::RZ;
    for (size_t i = 0; i < values.size(); i += 2) {
      Value lo = values[i];
      Value hi =
          i + 1 < values.size() ? values[i + 1] : llvmBuilder.f32_val(0.0);
      Value converted =
          NVVM::ConvertF32x2ToBF16x2Op::create(
              rewriter, loc, bf16x2Ty, hi, lo, Value(), nvvmRounding,
              NVVM::SaturationMode::NONE, /*relu=*/false)
              .getDst();
      result.push_back(llvmBuilder.extract_element(bf16_ty, converted,
                                                   llvmBuilder.i32_val(0)));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.extract_element(bf16_ty, converted,
                                                     llvmBuilder.i32_val(1)));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp16ToFp8(Location loc, ConversionPatternRewriter &rewriter,
                   ArrayRef<Value> values, Type fp8Ty) {
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    Type f16x2Ty = VectorType::get({2}, f16_ty);
    for (size_t i = 0; i < values.size(); i += 2) {
      Value lo = values[i];
      Value hi =
          i + 1 < values.size() ? values[i + 1] : llvmBuilder.f16_val(0.0);
      Value packedF16 = llvmBuilder.undef(f16x2Ty);
      // IFWA consumes the packed F16 source in logical lane order: vector
      // element 0 produces the low FP8 byte and element 1 the high byte.
      packedF16 = llvmBuilder.insert_element(f16x2Ty, packedF16, lo,
                                             llvmBuilder.i32_val(0));
      packedF16 = llvmBuilder.insert_element(f16x2Ty, packedF16, hi,
                                             llvmBuilder.i32_val(1));
      Value packedFp8 =
          NVVM::ConvertF16x2ToF8x2Op::create(rewriter, loc, i16_ty, packedF16,
                                             /*relu=*/false, fp8Ty)
              .getDst();
      Value packedI32 = llvmBuilder.zext(i32_ty, packedFp8);
      result.push_back(llvmBuilder.trunc(i8_ty, packedI32));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.trunc(
            i8_ty, llvmBuilder.lshr(packedI32, llvmBuilder.i32_val(8))));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp32ToFp8(Location loc, ConversionPatternRewriter &rewriter,
                   ArrayRef<Value> values, Type fp8Ty) {
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    for (size_t i = 0; i < values.size(); i += 2) {
      Value lo = values[i];
      Value hi =
          i + 1 < values.size() ? values[i + 1] : llvmBuilder.f32_val(0.0);
      // The NVVM op maps operand a to the upper output byte and operand b to
      // the lower output byte.
      Value packed =
          NVVM::ConvertF32x2ToF8x2Op::create(
              rewriter, loc, i16_ty, hi, lo, NVVM::FPRoundingMode::RN,
              NVVM::SaturationMode::SATFINITE, /*relu=*/false, fp8Ty)
              .getDst();
      Value packedI32 = llvmBuilder.zext(i32_ty, packed);
      result.push_back(llvmBuilder.trunc(i8_ty, packedI32));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.trunc(
            i8_ty, llvmBuilder.lshr(packedI32, llvmBuilder.i32_val(8))));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp8ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                   ArrayRef<Value> values, Type fp8Ty) {
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    Type fp8x2Ty = VectorType::get({2}, i8_ty);
    Type f16x2Ty = VectorType::get({2}, f16_ty);
    for (size_t i = 0; i < values.size(); i += 2) {
      Value hi = i + 1 < values.size() ? values[i + 1] : llvmBuilder.i8_val(0);
      Value packedFp8 = llvmBuilder.undef(fp8x2Ty);
      packedFp8 = llvmBuilder.insert_element(fp8x2Ty, packedFp8, values[i],
                                             llvmBuilder.i32_val(0));
      packedFp8 = llvmBuilder.insert_element(fp8x2Ty, packedFp8, hi,
                                             llvmBuilder.i32_val(1));
      Value converted =
          NVVM::ConvertF8x2ToF16x2Op::create(rewriter, loc, f16x2Ty, packedFp8,
                                             /*relu=*/false, fp8Ty)
              .getDst();
      result.push_back(llvmBuilder.extract_element(f16_ty, converted,
                                                   llvmBuilder.i32_val(0)));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.extract_element(f16_ty, converted,
                                                     llvmBuilder.i32_val(1)));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp16ToBf16InlineAsm(Location loc, ConversionPatternRewriter &rewriter,
                             ArrayRef<Value> values) {
    auto *ctx = rewriter.getContext();
    auto llvmBuilder = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result;
    Type bf16x2Ty = vec_ty(bf16_ty, 2);
    for (size_t i = 0; i < values.size(); i += 2) {
      Value lo = values[i];
      Value hi =
          i + 1 < values.size() ? values[i + 1] : llvmBuilder.f16_val(0.0);
      // PTX exposes scalar F16 -> BF16 conversion. Keep both conversions in
      // one inline-asm op and pack the two 16-bit results into a BF16x2 word.
      Value converted =
          LLVM::InlineAsmOp::create(
              rewriter, loc, i32_ty, ValueRange{lo, hi},
              "{\n"
              "  .reg .b16 b<2>;\n"
              "  cvt.rn.bf16.f16 b0, $1;\n"
              "  cvt.rn.bf16.f16 b1, $2;\n"
              "  mov.b32 $0, {b0, b1};\n"
              "}",
              "=r,h,h",
              /*has_side_effects=*/false,
              /*is_align_stack=*/false, LLVM::TailCallKind::None,
              LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
              ArrayAttr())
              .getRes();
      Value convertedVec = llvmBuilder.bitcast(converted, bf16x2Ty);
      result.push_back(llvmBuilder.extract_element(bf16_ty, convertedVec,
                                                   llvmBuilder.i32_val(0)));
      if (i + 1 < values.size())
        result.push_back(llvmBuilder.extract_element(bf16_ty, convertedVec,
                                                     llvmBuilder.i32_val(1)));
    }
    return result;
  }

  static SmallVector<Value>
  convertFp8ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                   ArrayRef<Value> values, Type fp8Ty) {
    SmallVector<Value> fp16Values =
        convertFp8ToFp16(loc, rewriter, values, fp8Ty);
    // Keep the NVVM representation here. The IFWA backend combines FP8 -> F16
    // followed by FPExt into a native FP8 -> FP32 conversion instruction.
    return llvm::to_vector(llvm::map_range(fp16Values, [&](Value value) {
      return LLVM::FPExtOp::create(rewriter, loc, f32_ty, value).getResult();
    }));
  }

  static SmallVector<Value>
  convertFp8E4M3FNToBf16(Location loc, ConversionPatternRewriter &rewriter,
                         ArrayRef<Value> values) {
    SmallVector<Value> fp16Values = convertFp8ToFp16(
        loc, rewriter, values, Float8E4M3FNType::get(rewriter.getContext()));
    return convertFp16ToBf16InlineAsm(loc, rewriter, fp16Values);
  }

  static SmallVector<Value>
  convertFp32ToFp8E5M2RTZ(Location loc, ConversionPatternRewriter &rewriter,
                          ArrayRef<Value> values) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto halves = convertFp32ToFp16(loc, rewriter, values, RoundingMode::RTZ);
    // Extract each half's high byte after F32 -> F16 RZ. This is not
    // satfinite and must not be replaced with the native RN packed conversion.
    return llvm::to_vector(llvm::map_range(halves, [&](Value value) -> Value {
      Value bits = b.zext(i32_ty, b.bitcast(value, i16_ty));
      return b.trunc(i8_ty, b.lshr(bits, b.i32_val(8)));
    }));
  }

  static SmallVector<Value>
  convertFp8E5M2ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                       ArrayRef<Value> values) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Reinterpret the sign/exponent/mantissa as BF16, then multiply by 2^112.
    // Express the exact scaling in F32 LLVM ops instead of packed PTX BF16
    // multiplication.
    // All intermediate/results are exactly BF16-representable, including
    // subnormals. Treat exponent 31 as finite:
    // E5M2 0x7c..0x7f become 65536, 81920, 98304, 114688 (sign symmetric).
    return llvm::to_vector(llvm::map_range(values, [&](Value value) {
      Value bits = b.zext(i32_ty, value);
      Value magnitude = b.shl(b.and_(bits, b.i32_val(0x7f)), b.i32_val(21));
      Value sign = b.shl(b.and_(bits, b.i32_val(0x80)), b.i32_val(24));
      Value expanded = b.bitcast(b.or_(magnitude, sign), f32_ty);
      Value scale = b.bitcast(b.i32_val(0x77800000), f32_ty);
      Value scaled = LLVM::FMulOp::create(rewriter, loc, expanded, scale);
      return LLVM::FPTruncOp::create(rewriter, loc, bf16_ty, scaled)
          .getResult();
    }));
  }

  SmallVector<Value> createDestOps(FpToFpOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type srcTy = getElementTypeOrSelf(op.getSrc());
    Type dstTy = getElementTypeOrSelf(op.getResult());
    SmallVector<Value> values = getConversionPair(operands);
    if (isa<Float16Type, BFloat16Type>(srcTy) && dstTy.isF32())
      return llvm::to_vector(llvm::map_range(values, [&](Value value) {
        return LLVM::FPExtOp::create(rewriter, loc, f32_ty, value).getResult();
      }));
    if (srcTy.isF32() && dstTy.isF16())
      return convertFp32ToFp16(loc, rewriter, values, *op.getRounding());
    if (srcTy.isF32() && dstTy.isBF16())
      return convertFp32ToBf16(loc, rewriter, values, *op.getRounding());
    if (isa<Float8E4M3FNType, Float8E5M2Type>(srcTy) && dstTy.isF16())
      return convertFp8ToFp16(loc, rewriter, values, srcTy);
    if (isa<Float8E4M3FNType, Float8E5M2Type>(srcTy) && dstTy.isF32())
      return convertFp8ToFp32(loc, rewriter, values, srcTy);
    if (isa<Float8E5M2Type>(srcTy) && dstTy.isBF16())
      return convertFp8E5M2ToBf16(loc, rewriter, values);
    if (isa<Float8E4M3FNType>(srcTy) && dstTy.isBF16())
      return convertFp8E4M3FNToBf16(loc, rewriter, values);
    if (srcTy.isF16())
      return convertFp16ToFp8(loc, rewriter, values, dstTy);

    if (srcTy.isF32() && op.getRounding() == RoundingMode::RTZ)
      return convertFp32ToFp8E5M2RTZ(loc, rewriter, values);

    SmallVector<Value> fp32Values =
        llvm::to_vector(llvm::map_range(values, [&](Value value) -> Value {
          if (srcTy.isBF16())
            return LLVM::FPExtOp::create(rewriter, loc, f32_ty, value);
          return value;
        }));
    return convertFp32ToFp8(loc, rewriter, fp32Values, dstTy);
  }
};

void populateGPNPUElementwiseToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  // IFWA-specific AddPtr pattern.
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);

  // Reuse base elementwise patterns for integer ops, comparisons, select, etc.
  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  patterns.add<SymmetricClampFOpConversion>(
      typeConverter, axisInfoAnalysis,
      mlir::triton::patternBenefitClampOptimizedPattern);

  // These float operations are target-specific.
#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseToIntrinsicOpConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "llvm.nvvm.div.rn.f", benefit);
  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);
  POPULATE_OP(arith::NegFOp, LLVM::FNegOp);
  POPULATE_OP(arith::SIToFPOp, LLVM::SIToFPOp);
  POPULATE_OP(arith::UIToFPOp, LLVM::UIToFPOp);
  POPULATE_OP(arith::FPToSIOp, LLVM::FPToSIOp);
  POPULATE_OP(arith::FPToUIOp, LLVM::FPToUIOp);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, benefit);
#undef POPULATE_OP
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
