#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::getElementType;
using mlir::triton::gpu::getFunctionType;
using mlir::triton::gpu::MultipleOperandsRange;

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

namespace {
// ROCM utility functions for data type conversion
/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format
// NVIDIA GPU supports it natively but we don't have hardware native
// support on MI300.
// The SW based downcast with RTNE is not fully functional for the
// denorm values. We need rewrite it if we need to emulate this data type
// on HCUGPU.
static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  Value a0 = bitcast(fp16x2Vec0, i32_ty);
  Value a1 = bitcast(fp16x2Vec1, i32_ty);

  a0 = and_(i32_ty, a0, i32_val(0xfffefffe));
  a1 = and_(i32_ty, a1, i32_val(0xfffefffe));

  a0 = add(i32_ty, a0, i32_val(0x00800080));
  a1 = add(i32_ty, a1, i32_val(0x00800080));

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = bitcast(a0, fp8x4VecTy);
  a1 = bitcast(a1, fp8x4VecTy);

  return {extract_element(i8_ty, a0, i32_val(1)),
          extract_element(i8_ty, a0, i32_val(3)),
          extract_element(i8_ty, a1, i32_val(1)),
          extract_element(i8_ty, a1, i32_val(3))};
}

static SmallVector<Value>
Fp16_to_Fp8E5M2_RTZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  Value a0 = bitcast(fp16x2Vec0, i32_ty);
  Value a1 = bitcast(fp16x2Vec1, i32_ty);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = bitcast(a0, fp8x4VecTy);
  a1 = bitcast(a1, fp8x4VecTy);

  return {extract_element(i8_ty, a0, i32_val(1)),
          extract_element(i8_ty, a0, i32_val(3)),
          extract_element(i8_ty, a1, i32_val(1)),
          extract_element(i8_ty, a1, i32_val(3))};
}

//===----------------===//
///      FP8E4M3
//===----------------===//
static Value
Fp32_to_Fp8E4M3FN_RTNE_oneValue(Location loc,
                                ConversionPatternRewriter &rewriter, Value v) {
  // == Step 1: Bitcast to i32 and extract components ==
  Value vi32 = bitcast(v, i32_ty);
  Value sign = trunc(i8_ty, lshr(vi32, i32_val(24))); // get sign << 7
  sign = and_(i8_ty, sign, i8_val(0x80));             // keep sign bit

  Value abs = and_(i32_ty, vi32, i32_val(0x7FFFFFFF)); // strip sign

  // == Step 2: NaN check ==
  Value isNaN = and_(
      i1_ty,
      icmp_eq(and_(i32_ty, vi32, i32_val(0x7F800000)),
              i32_val(0x7F800000)), // exp == 0xFF
      icmp_ne(and_(i32_ty, vi32, i32_val(0x007FFFFF)), i32_val(0)) // frac != 0
  );

  // == Step 3: Rounding (RTNE) ==
  // bias diff = (127 - 7) << 23 = 120 << 23 = 0x3C000000
  // mantissa alignment: keep top 3 mantissa bits => shift right by 23 - 3 = 20
  constexpr uint32_t baseRoundingBias =
      (1 << 19) - 1; // 0x7FFFF = round-to-even

  Value roundBit = lshr(and_(i32_ty, vi32, i32_val(0x00100000)),
                        i32_val(20)); // bit 20 (for even)
  Value roundingBias = add(i32_val(baseRoundingBias), roundBit);
  Value vFp8 = add(vi32, roundingBias);

  // Keep top 9 bits (sign | exp(4) | mant(3)) ← trunc later
  vFp8 = and_(i32_ty, vFp8, i32_val(0xFFFFFF80)); // clear bottom bits

  // == Step 4: Clamp to min normal (smallest FP8 normal in FP32) ==
  // FP32 representation of 2^-6 = 2.0^(-6) = 0x38800000
  vFp8 = select(icmp_ult(vFp8, i32_val(0x38800000)), i32_val(0x38800000), vFp8);

  // == Step 5: Adjust exponent bias ==
  // Subtract (127 - 7) << 23 = 0x3C000000
  vFp8 = sub(vFp8, i32_val(0x3C000000));

  // Shift right to extract FP8 bits: (exp + mant) → shift 20 to fit 3-bit mant
  vFp8 = trunc(i8_ty, lshr(vFp8, i32_val(20)));

  // == Step 6: Clamp to FP8 max value (before inf) ==
  // 0x7E is largest finite E4M3FN value (0.1111.10x)
  Value isOverflowOrInf =
      icmp_ugt(abs, i32_val(0x7F7FFFFF)); // > max finite FP32
  vFp8 = select(isOverflowOrInf, i8_val(0x7E), vFp8);

  // == Step 7: Handle subnormals via LUT ==
  constexpr size_t lutSize = 8;
  constexpr uint32_t halfwayPointsLUT[lutSize] = {
      0x33800000, 0x37000000, 0x38800000, 0x39400000,
      0x3A000000, 0x3A800000, 0x3B000000, 0x3B800000};

  for (int i = lutSize - 1; i >= 0; --i) {
    Value cmp;
    if (i % 2 == 0) {
      cmp = icmp_ule(abs, i32_val(halfwayPointsLUT[i]));
    } else {
      cmp = icmp_ult(abs, i32_val(halfwayPointsLUT[i]));
    }
    vFp8 = select(cmp, i8_val(i), vFp8);
  }

  // == Step 8: Handle NaN ==
  vFp8 = select(isNaN, i8_val(0x7F), vFp8);

  // == Step 9: Set sign bit ==
  vFp8 = or_(vFp8, sign);
  return vFp8;
}

static SmallVector<Value>
Fp32_to_Fp8E4M3FN_RTNE(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  assert(v.size() == 4 && "Expected 4 values for FP8E4M3FN conversion");
  SmallVector<Value> result(4);
  for (size_t i = 0; i < 4; ++i) {
    result[i] = Fp32_to_Fp8E4M3FN_RTNE_oneValue(loc, rewriter, v[i]);
  }
  return result;
}

// Cast FP16 to FP8E4M3FN in saturation and round-to-nearest-even mode.
// According to
// https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1,
// In saturation mode, inf and out-of-range numbers are converted to the largest
// normal number, i.e. ±448. NaNs are converted to NaNs.
static Value
Fp16_to_Fp8E4M3FN_RTNE_oneValue(Location loc,
                                ConversionPatternRewriter &rewriter, Value v) {
  // StringRef funcName = "llvm.is.fpclass";
  // Value isNaN = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, funcName,
  // i1_ty,
  //                                               {v, i32_val(0x3)})
  //                   ->getResult(0);
  // Get sign and absolute value
  Value vi16 = bitcast(v, i16_ty);
  Value isNaN = and_(
      icmp_eq(and_(vi16, i16_val(0x7C00)), i16_val(0x7C00)), //; exp == 0x7C00
      icmp_ne(and_(vi16, i16_val(0x03FF)), i16_val(0))       //; frac != 0
  );

  Value sign = trunc(i8_ty, lshr(and_(vi16, i16_val(0x8000)), i16_val(8)));
  vi16 = and_(vi16, i16_val(0x7FFF));

  // Rounding to nearest even
  constexpr uint16_t baseRoundingBias = 0x003F; // 1 << (10 - 3 - 1) - 1

  // S.EEEEE.MMMMMMMMMM => 0.00000.00M0000000 => 0.00000.000000000M
  Value remainingMantissaLSB = lshr(and_(vi16, i16_val(0x0080)), i16_val(7));
  Value roundingBias = add(remainingMantissaLSB, i16_val(baseRoundingBias));
  Value vFp8 = add(vi16, roundingBias);

  // Reduce mantissa to 3 bits
  vFp8 = and_(vFp8, i16_val(0xFF80)); // 0xFF80 == 1.11111.1110000000

  // 0x2400 is the FP16 representation of 2^{-6}, which is the smallest normal
  // number in FP8E4M3FN. We round numbers smaller than that to 0x2400 to make
  // it easier to handle subnormals
  vFp8 = umax(vFp8, i16_val(0x2400));

  // Adjust exponent bias
  vFp8 = sub(vFp8, i16_val(0x2000)); // (15 - 7) << 10

  // Shift right and truncate
  vFp8 = trunc(i8_ty, lshr(vFp8, i16_val(7))); // 10 - 3

  // 0x5F7F == 0.10111.1101111111 is the largest possible normal
  // number(including infinity) after rounding in FP8
  //
  // In saturation mode, numbers larger than the max normal number(including
  // infinity) in FP8 after rounding will be replaced with max_E4M3, i.e. 0x7E
  // === 0.1111.110
  Value isOverflowOrInf = icmp_ugt(vi16, i16_val(0x5F7F));
  vFp8 = select(isOverflowOrInf, i8_val(0x7E), vFp8);

  // Round subnormals to nearest even. Ref:
  // https://github.com/openxla/xla/blob/f20c6fe2/xla/service/elemental_ir_emitter.cc#L272
  constexpr size_t lutSize = 8;
  constexpr float halfwayPointsLUT[lutSize] = {0x1400, 0x1A00, 0x1D00, 0x1F00,
                                               0x2080, 0x2180, 0x2280, 0x2380};

  for (int i = lutSize - 1; i >= 0; i--) {
    Value cmp;
    if (i % 2 == 0) {
      cmp = icmp_ule(vi16, i16_val(halfwayPointsLUT[i]));
    } else {
      cmp = icmp_ult(vi16, i16_val(halfwayPointsLUT[i]));
    }

    vFp8 = select(cmp, i8_val(i), vFp8);
  }

  // NaN remains NaN after conversion
  vFp8 = select(isNaN, i8_val(0x7F), vFp8);

  // Set sign bit
  vFp8 = or_(i8_ty, vFp8, sign);

  return vFp8;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FN_RTNE(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = Fp16_to_Fp8E4M3FN_RTNE_oneValue(loc, rewriter, v[i]);
  }
  return result;
}

static Value cvtFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v) {
  GCNBuilder builder;
  auto &cvt = *builder.create("v_cvt_f32_f16");
  auto res = builder.newOperand("=v");
  auto operand = builder.newOperand(v, "v");
  cvt(res, operand);
  return builder.launch(rewriter, loc, f32_ty, false);
}

static Value cvtFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v, const RoundingMode rounding) {
  GCNBuilder builder;

  auto &cvt = *builder.create("v_cvt_f16_f32");
  auto res = builder.newOperand("=v");
  auto operand = builder.newOperand(v, "v");
  if (rounding == RoundingMode::RTZ) {
    auto &setRTZ = *builder.create("s_setreg_imm32_b32 0x1801, 0xc");
    setRTZ();
  }
  cvt(res, operand);
  if (rounding == RoundingMode::RTZ) {
    auto &resetRTZ = *builder.create("s_setreg_imm32_b32 0x1801, 0x0");
    resetRTZ();
  }
  return builder.launch(rewriter, loc, f16_ty, false);
}

// convert fp8 to fp32
static SmallVector<Value> cvtFp8ToFp32(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v0, Value v1,
                                       const std::string &fp8_format) {
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_f32_" + fp8_format;

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = undef(fp8x4VecTy);
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
  auto i32v = bitcast(fp8x4Vec, i32_ty);

  GCNBuilder builder1;
  auto &cvt = *builder1.create(ins_str);
  auto res = builder1.newOperand("=v");
  auto operand = builder1.newOperand(i32v, "v");
  cvt(res, operand);
  auto i64v = builder1.launch(rewriter, loc, i64_ty, false);
  auto fp32x2VecTy = vec_ty(f32_ty, 2);
  auto fp32x2Vec = bitcast(i64v, fp32x2VecTy);

  SmallVector<Value> ret(2);
  ret[0] = extract_element(f32_ty, fp32x2Vec, i32_val(0));
  ret[1] = extract_element(f32_ty, fp32x2Vec, i32_val(1));

  return ret;
}

// convert fp32 to fp8
static SmallVector<Value> cvtFp32ToFp8(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v0, Value v1,
                                       const std::string &fp8_format) {
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_" + fp8_format + "_f32";

  GCNBuilder builder;
  auto &cvt = *builder.create(ins_str);
  auto res = builder.newOperand("=v");
  auto operand0 = builder.newOperand(v0, "v");
  auto operand1 = builder.newOperand(v1, "v");
  cvt(res, operand0, operand1);
  auto fp8x4Vec = builder.launch(rewriter, loc, i32_ty, false);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  auto a1 = bitcast(fp8x4Vec, fp8x4VecTy);

  SmallVector<Value> ret(2);
  ret[0] = extract_element(i8_ty, a1, i32_val(0));
  ret[1] = extract_element(i8_ty, a1, i32_val(1));

  return ret;
}
static SmallVector<Value>
convert_val_Fp16_to_Fp8(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1, const std::string &fp8_format) {
  assert(fp8_format == "fp8" || fp8_format == "bf8");
  std::string ins_str = "v_cvt_pk_" + fp8_format + "_f32";

  auto f32_0 = cvtFp16ToFp32(loc, rewriter, v0);
  auto f32_1 = cvtFp16ToFp32(loc, rewriter, v1);

  // Convert fp32 to fp8
  return cvtFp32ToFp8(loc, rewriter, f32_0, f32_1, fp8_format);
}

static SmallVector<Value>
convert_val_Fp8_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v0, Value v1, const std::string &fp8_format) {

  // Convert fp8 to fp32
  SmallVector<Value> ret = cvtFp8ToFp32(loc, rewriter, v0, v1, fp8_format);

  // Convert fp32 to fp16
  ret[0] = cvtFp32ToFp16(loc, rewriter, ret[0], RoundingMode::RTNE);
  ret[1] = cvtFp32ToFp16(loc, rewriter, ret[1], RoundingMode::RTNE);

  return ret;
}

static SmallVector<Value>
Fp32_to_Fp8E5M2FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp32ToFp8(loc, rewriter, v[0], v[1], "bf8");
}

static SmallVector<Value>
Fp32_to_Fp8E4M3FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp32ToFp8(loc, rewriter, v[0], v[1], "fp8");
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "bf8");
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
}
// Depend on whether we focus more on performance, we may skip
// the processing of submornal values
static Value Fp16_to_Fp8E5M2FNUZ_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto vi16 = bitcast(v, i16_ty);
  auto e = and_(i16_ty, vi16, int_val(16, 0x7C00));
  auto sign = and_(i16_ty, vi16, int_val(16, 0x8000));

  // normal value
  auto a = and_(i16_ty, vi16, int_val(16, 0x7FFFF));
  auto a1 = add(i16_ty, a, int_val(16, 0x0400));
  auto o1 = or_(i16_ty, a1, sign);

  // subnormal value, e is 0
  auto m = and_(i16_ty, vi16, int_val(16, 0x03FF));
  auto m2 = shl(m, int_val(16, 1));
  auto o2 = or_(i16_ty, sign, or_(i16_ty, int_val(16, 1), m2));

  auto e_is_zero = icmp_eq(e, int_val(16, 0));
  auto e_is_all1 = icmp_eq(e, int_val(16, 0x7C00));

  auto ot = select(e_is_zero, o2, o1);
  auto o = select(e_is_all1, vi16, ot);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = bitcast(o, fp8x2VecTy);

  return extract_element(i8_ty, res, i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp16_to_Fp8E5M2FNUZ_oneValue(loc, rewriter, v[0]);
  result[1] = Fp16_to_Fp8E5M2FNUZ_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E5M2FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp16_to_Fp8(loc, rewriter, v[0], v[1], "bf8");
}

ConverterT Fp16_to_Fp8E5M2FNUZ(HCU::ISAFamily isaFamily) {
  return isaFamily == HCU::ISAFamily::CDNA3 ? Fp16_to_Fp8E5M2FNUZ_HW
                                            : Fp16_to_Fp8E5M2FNUZ_SW;
}

static Value Fp8E4M3FN_to_Fp32_oneValue(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Value v) {
  // Bitcast FP8 (i8) value
  Value vi8 = bitcast(v, i8_ty);

  // Extract sign and absolute value
  Value sign = and_(vi8, i8_val(0x80)); // 0b1000'0000
  Value vAbs = and_(vi8, i8_val(0x7F)); // 0b0111'1111

  // Extract exponent and mantissa
  Value exp = and_(vAbs, i8_val(0x78));  // 0b0111'1000
  Value mant = and_(vAbs, i8_val(0x07)); // 0b0000'0111

  // Right shift exponent to LSB
  exp = lshr(exp, i8_val(3));

  // Detect special cases
  Value isZeroOrDenorm = icmp_ult(vAbs, i8_val(0x08)); // < 0b0000'1000
  Value isNaN = icmp_eq(vAbs, i8_val(0x7F));           // == 0b0111'1111

  // Default normal conversion
  // Compose 32-bit float:
  // sign << 31 | (exp + 120) << 23 | (mant << 20)
  Value sign32 = shl(zext(i32_ty, sign), i32_val(24)); // move sign to bit 31
  Value exp32 = shl(zext(i32_ty, exp), i32_val(23));   // move exp to bit 23
  Value expBias = i32_val(120 << 23);                  // bias diff (127 - 7)
  exp32 = add(exp32, expBias);

  Value mant32 =
      shl(zext(i32_ty, mant), i32_val(20)); // move mant to bits 22-20

  Value combined = or_(or_(sign32, exp32), mant32);

  // Handle NaN: output canonical FP32 NaN 0x7FC00000
  Value nanVal = i32_val(0x7FC00000);
  combined = select(isNaN, nanVal, combined);

  // Handle denorm/zero via LUT
  // For vAbs in [0x00 ~ 0x07] → use LUT
  constexpr int lutSize = 8;
  static constexpr uint32_t denormsAndZeroLut[lutSize] = {
      0x00000000, 0x38800000, 0x39800000, 0x3A000000, 0x3A800000,
      0x3B000000, 0x3B400000, 0x3B800000}; // approximated FP32 tiny values

  for (int i = 0; i < lutSize; ++i) {
    combined = select(icmp_eq(vAbs, i8_val(i)), i32_val(denormsAndZeroLut[i]),
                      combined);
  }

  // Bitcast to float
  Value result = bitcast(combined, f32_ty);
  return result;
}

static SmallVector<Value> Fp8E4M3FN_to_Fp32(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &values) {
  SmallVector<Value> results(values.size());
  for (size_t i = 0; i < values.size(); i++)
    results[i] = Fp8E4M3FN_to_Fp32_oneValue(loc, rewriter, values[i]);
  return results;
}

static Value Fp8E4M3FN_to_Fp16_oneValue(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Value v) {
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = undef(fp8x2VecTy);
  a = insert_element(fp8x2VecTy, a, i8_val(0), i32_val(0));
  a = insert_element(fp8x2VecTy, a, v, i32_val(1));
  a = bitcast(a, i16_ty);

  // Get sign and absolute value
  Value sign = and_(a, i16_val(0x8000));
  a = and_(a, i16_val(0x7FFF));

  // Right shift 1 bit to adjust the positions of exponent and mantissa
  a = lshr(a, i16_val(1));

  // Adjust exponent, (15 - 7) << 10 === 0x2000
  a = add(a, i16_val(0x2000));

  // Check NaN
  Value vAbs = and_(bitcast(v, i8_ty), i8_val(0x7F));
  a = select(icmp_eq(vAbs, i8_val(0x7F)), i16_val(0x7E00), a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  constexpr size_t lutSize = 8;
  static constexpr int denormsAndZeroLut[lutSize] = {
      0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};

  for (int i = 0; i < lutSize; i++) {
    a = select(icmp_eq(vAbs, i8_val(i)), i16_val(denormsAndZeroLut[i]), a);
  }

  // Set sign
  a = or_(a, sign);
  a = bitcast(a, f16_ty);

  return a;
}

static SmallVector<Value> Fp8E4M3FN_to_Fp16(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &values) {
  SmallVector<Value> results(values.size());
  for (size_t i = 0; i < values.size(); i++)
    results[i] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[i]);
  return results;
}

static SmallVector<Value> Fp8E5M2_to_Fp16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);
  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = bitcast(a1, fp16x2VecTy);

  return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
}

static Value convertBf16ToFp32(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v) {
  auto as_int16 = bitcast(v, i16_ty);
  auto as_int32 = zext(i32_ty, as_int16);
  auto shifted = shl(i32_ty, as_int32, i32_val(16));
  return bitcast(shifted, f32_ty);
}

static Value convertFp32ToBf16(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v, const RoundingMode rounding) {
  if (rounding == RoundingMode::RTZ) {
    auto as_int32 = bitcast(v, i32_ty);
    auto shifted = lshr(i32_ty, as_int32, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    return bitcast(truncated, bf16_ty);
  }
  // Otherwise it is (rounding == RoundingMode::RTNE)
  auto as_uint32 = bitcast(v, i32_ty);
  auto check_exponent =
      and_(i32_ty, xor_(i32_ty, as_uint32, i32_val(0xffffffff)),
           i32_val(0x7f800000));
  auto exponent_not_all1s = icmp_ne(check_exponent, i32_val(0));
  auto exponent_all1s = icmp_eq(check_exponent, i32_val(0));
  auto rounded =
      add(i32_ty, i32_val(0x7fff),
          and_(i32_ty, lshr(i32_ty, as_uint32, i32_val(16)), i32_val(1)));
  rounded = add(i32_ty, rounded, as_uint32);
  auto res = select(exponent_not_all1s, rounded, as_uint32);

  auto preserve_nan =
      and_(i1_ty, exponent_all1s,
           icmp_ne(and_(i32_ty, as_uint32, i32_val(0xffff)), i32_val(0)));
  auto nan = or_(i32_ty, as_uint32, i32_val(0x10000));
  res = select(preserve_nan, nan, res);

  auto shifted = lshr(i32_ty, res, i32_val(16));
  auto truncated = trunc(i16_ty, shifted);
  return bitcast(truncated, bf16_ty);
}

static Value Fp8E5M2FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = undef(fp8x2VecTy);
  a = insert_element(fp8x2VecTy, a, int_val(8, 0), i32_val(0));
  a = insert_element(fp8x2VecTy, a, v, i32_val(1));
  a = bitcast(a, i16_ty);

  auto e = and_(i16_ty, a, int_val(16, 0x7C00));
  auto m = and_(i16_ty, a, int_val(16, 0x0300));
  auto sign = and_(i16_ty, a, int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = icmp_eq(e, int_val(16, 0x0));

  // case 1, e is zero, need to move m right by 1 bit
  auto m1 = lshr(i16_ty, m, int_val(16, 1));
  auto o0 = or_(i16_ty, sign, m1);

  // case 2, e is nonzero, sub exponent by 1
  auto e1 = sub(i16_ty, e, int_val(16, 0x0400));

  auto e_is_one = icmp_eq(e, int_val(16, 0x0400));
  auto m2 = add(i16_ty, m1, int_val(16, 0x0200));

  auto o1 = or_(i16_ty, sign, or_(i16_ty, m, e1));
  auto o2 = or_(i16_ty, sign, m2);

  auto o12 = select(e_is_one, o2, o1);
  auto o = select(e_is_zero, o0, o12);

  return bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp8_to_Fp16(loc, rewriter, v[0], v[1], "bf8");
}

ConverterT Fp8E5M2FNUZ_to_Fp16(HCU::ISAFamily isaFamily) {
  return isaFamily == HCU::ISAFamily::CDNA3 ? Fp8E5M2FNUZ_to_Fp16_HW
                                            : Fp8E5M2FNUZ_to_Fp16_SW;
}

static SmallVector<Value> Fp8E5M2_to_Bf16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  b0 = lshr(i32_ty, b0, i32_val(3));
  b1 = lshr(i32_ty, b1, i32_val(3));

  Value c0 = shl(i32_ty, b0, i32_val(16));
  Value c1 = and_(i32_ty, b0, i32_val(0xFFFF0000));
  Value c2 = shl(i32_ty, b1, i32_val(16));
  Value c3 = and_(i32_ty, b1, i32_val(0xFFFF0000));

  c0 = bitcast(c0, f32_ty);
  c1 = bitcast(c1, f32_ty);
  c2 = bitcast(c2, f32_ty);
  c3 = bitcast(c3, f32_ty);

  Value d0 = fmul(f32_ty, c0, f32_val(0x1p+112));
  Value d1 = fmul(f32_ty, c1, f32_val(0x1p+112));
  Value d2 = fmul(f32_ty, c2, f32_val(0x1p+112));
  Value d3 = fmul(f32_ty, c3, f32_val(0x1p+112));

  d0 = bitcast(d0, i32_ty);
  d1 = bitcast(d1, i32_ty);
  d2 = bitcast(d2, i32_ty);
  d3 = bitcast(d3, i32_ty);

  Value out0 = or_(i32_ty, lshr(i32_ty, d0, i32_val(16)), d1);
  Value out1 = or_(i32_ty, lshr(i32_ty, d2, i32_val(16)), d3);

  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));

  out0 = or_(i32_ty, out0, sign0);
  out1 = or_(i32_ty, out1, sign1);

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  out0 = bitcast(out0, bf16x2VecTy);
  out1 = bitcast(out1, bf16x2VecTy);

  return {extract_element(bf16_ty, out0, i32_val(0)),
          extract_element(bf16_ty, out0, i32_val(1)),
          extract_element(bf16_ty, out1, i32_val(0)),
          extract_element(bf16_ty, out1, i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp8E5M2(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(1)), i32_val(0));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(3)), i32_val(1));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(1)), i32_val(2));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(3)), i32_val(3));
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x38000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x57e00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3800));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x57e0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x38000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x57e00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3800));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x57e0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x00100010));
  nosign1 = add(i32_ty, nosign1, i32_val(0x00100010));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x38003800));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x38003800));
  nosign0 = shl(i32_ty, nosign0, i32_val(3));
  nosign1 = shl(i32_ty, nosign1, i32_val(3));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(1)), i32_val(0));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(3)), i32_val(1));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(1)), i32_val(2));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(3)), i32_val(3));
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
          extract_element(i8_ty, fp8x4Vec, i32_val(1)),
          extract_element(i8_ty, fp8x4Vec, i32_val(2)),
          extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}

// ROCM type conversion between fp8 and bf16
// fp8e4m3fnuz to bf16
static SmallVector<Value>
Fp8E4M3FNUZ_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto ret = cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
  ret[0] = convertFp32ToBf16(loc, rewriter, ret[0], RoundingMode::RTZ);
  ret[1] = convertFp32ToBf16(loc, rewriter, ret[1], RoundingMode::RTZ);
  return ret;
}

// bf16 to fp8e4m3fnuz
static SmallVector<Value>
Bf16_to_Fp8E4M3FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto v0 = convertBf16ToFp32(loc, rewriter, v[0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, v[1]);
  return cvtFp32ToFp8(loc, rewriter, v0, v1, "fp8");
}

// fp8e5m2fnuz to bf16
static SmallVector<Value>
Fp8E5M2FNUZ_to_Bf16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto ret = cvtFp8ToFp32(loc, rewriter, v[0], v[1], "bf8");
  ret[0] = convertFp32ToBf16(loc, rewriter, ret[0], RoundingMode::RTZ);
  ret[1] = convertFp32ToBf16(loc, rewriter, ret[1], RoundingMode::RTZ);
  return ret;
}

// bf16 to fp8e5m2fnuz
static SmallVector<Value>
Bf16_to_Fp8E5M2FNUZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto v0 = convertBf16ToFp32(loc, rewriter, v[0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, v[1]);
  return cvtFp32ToFp8(loc, rewriter, v0, v1, "bf8");
}

static Value Fp8E4M3FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = undef(fp8x2VecTy);
  a = insert_element(fp8x2VecTy, a, int_val(8, 0), i32_val(0));
  a = insert_element(fp8x2VecTy, a, v, i32_val(1));
  a = bitcast(a, i16_ty);

  auto e_mask = int_val(16, 0x7A00);
  auto e = and_(i16_ty, a, e_mask);

  auto m = and_(i16_ty, a, int_val(16, 0x0700));
  auto sign = and_(i16_ty, a, int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = icmp_eq(e, int_val(16, 0x0));
  auto b = and_(i16_ty, a, int_val(16, 0x7FFF));
  auto b1 = lshr(i16_ty, b, int_val(16, 1));

  // case 1, e is nonzero, add exponent by 6
  auto o0v = add(i16_ty, b1, int_val(16, 0x0C00));
  auto o0 = or_(i16_ty, o0v, sign);

  // case 2, e is nonzero, add exponent by 7
  auto o1v = add(i16_ty, b1, int_val(16, 0x1C00));
  auto o1 = or_(i16_ty, o1v, sign);

  auto io = select(e_is_zero, o0, o1);
  return bitcast(io, f16_ty);
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp8_to_Fp16(loc, rewriter, v[0], v[1], "fp8");
}

static ConverterT Fp8E4M3FNUZ_to_Fp16(HCU::ISAFamily isaFamily) {
  return isaFamily == HCU::ISAFamily::CDNA3 ? Fp8E4M3FNUZ_to_Fp16_HW
                                            : Fp8E4M3FNUZ_to_Fp16_SW;
}

// Fp16 -> Fp8E4M3 (packed)
static Value Fp16_to_Fp8E4M3FNUZ_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto vi16 = bitcast(v, i16_ty);
  auto e10 = and_(vi16, int_val(16, 0x7C00));
  auto e = lshr(i16_ty, e10, int_val(16, 10));

  auto s = and_(i16_ty, vi16, int_val(16, 0x8000));

  auto m7 = and_(i16_ty, vi16, int_val(16, 0x0380));
  auto m = shl(i16_ty, m7, int_val(16, 1));

  // three cases:
  //  1) e > 21 --> e = 1111,
  //  2) e <= 7 ---> e = 0,
  //  3) others, normal conversion
  auto e1 = int_val(16, 0x7800);
  auto e2 = int_val(16, 0x0);
  auto e31 = sub(i16_ty, e10, int_val(16, 0x1C00));
  auto e3 = shl(i16_ty, e31, int_val(16, 1));

  auto c13 = icmp_sgt(e, int_val(16, 21));
  auto e13 = select(c13, e1, e3);
  auto c23 = icmp_sle(e, int_val(16, 7));
  auto re = select(c23, e2, e13);

  auto r = or_(i16_ty, s, or_(i16_ty, re, m));
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  auto res = bitcast(r, fp8x2VecTy);

  return extract_element(i8_ty, res, i32_val(1));
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  SmallVector<Value> result(2);
  result[0] = Fp16_to_Fp8E4M3FNUZ_oneValue(loc, rewriter, v[0]);
  result[1] = Fp16_to_Fp8E4M3FNUZ_oneValue(loc, rewriter, v[1]);

  return result;
}

static SmallVector<Value>
Fp16_to_Fp8E4M3FNUZ_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  return convert_val_Fp16_to_Fp8(loc, rewriter, v[0], v[1], "fp8");
}

static ConverterT Fp16_to_Fp8E4M3FNUZ(HCU::ISAFamily isaFamily) {
  return isaFamily == HCU::ISAFamily::CDNA3 ? Fp16_to_Fp8E4M3FNUZ_HW
                                            : Fp16_to_Fp8E4M3FNUZ_SW;
}

// WARN: subnormal (0bs0000xxx) are not handled
static SmallVector<Value> Fp8E4M3_to_Bf16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  b0 = lshr(i32_ty, b0, i32_val(4));
  b1 = lshr(i32_ty, b1, i32_val(4));

  b0 = add(i32_ty, b0, i32_val(0x3c003c00));
  b1 = add(i32_ty, b1, i32_val(0x3c003c00));
  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, b0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, b1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return {extract_element(bf16_ty, bf16x2Vec0, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec0, i32_val(1)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp8E4M3(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(1)), i32_val(0));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(3)), i32_val(1));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(1)), i32_val(2));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(3)), i32_val(3));
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x3c000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x43f00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3c00));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x43f0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x3c000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x43f00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3c00));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x43f0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x80008));
  nosign1 = add(i32_ty, nosign1, i32_val(0x80008));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x3c003c00));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x3c003c00));
  nosign0 = lshr(i32_ty, nosign0, i32_val(4));
  nosign1 = lshr(i32_ty, nosign1, i32_val(4));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(0)), i32_val(0));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(2)), i32_val(1));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(0)), i32_val(2));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(2)), i32_val(3));
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
          extract_element(i8_ty, fp8x4Vec, i32_val(1)),
          extract_element(i8_ty, fp8x4Vec, i32_val(2)),
          extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      triton::FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              HCU::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return cvtFp16ToFp32(loc, rewriter, v);
  }

  mlir::FailureOr<ConverterT>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<mlir::Float8E4M3B11FNUZType>();
    auto F8E4M3FNUZTyID = TypeID::get<mlir::Float8E4M3FNUZType>();
    auto F8E5M2FNUZTyID = TypeID::get<mlir::Float8E5M2FNUZType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F8E4M3FNTyID = TypeID::get<mlir::Float8E4M3FNType>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, ConverterT>
        srcMap = {
            // F8 -> F16
            {{F8E4M3FNUZTyID, F16TyID, undefRounding},
             Fp8E4M3FNUZ_to_Fp16(isaFamily)},
            {{F8E4M3FNTyID, F16TyID, undefRounding}, Fp8E4M3FN_to_Fp16},
            {{F8E5M2FNUZTyID, F16TyID, undefRounding},
             Fp8E5M2FNUZ_to_Fp16(isaFamily)},
            {{F8E5M2TyID, F16TyID, undefRounding}, Fp8E5M2_to_Fp16},
            // F16 -> F8
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FN_RTNE},
            {{F16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2FNUZ(isaFamily)},
            {{F16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FNUZ(isaFamily)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE}, Fp16_to_Fp8E5M2_RTNE},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding}, Fp8E5M2_to_Bf16},
            {{F8E5M2FNUZTyID, BF16TyID, undefRounding}, Fp8E5M2FNUZ_to_Bf16},
            {{F8E4M3FNUZTyID, BF16TyID, undefRounding}, Fp8E4M3FNUZ_to_Bf16},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE}, Bf16_to_Fp8E5M2},
            {{BF16TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2FNUZ},
            {{BF16TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3FNUZ},

            // F32 <-> F8
            {{F32TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FNUZ},
            {{F32TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2FNUZ},
            {{F32TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FN_RTNE},
            {{F8E4M3FNUZTyID, F32TyID, undefRounding}, Fp8E4M3FNUZ_to_Fp32},
            {{F8E5M2FNUZTyID, F32TyID, undefRounding}, Fp8E5M2FNUZ_to_Fp32},
            {{F8E4M3FNTyID, F32TyID, RoundingMode::RTNE}, Fp8E4M3FN_to_Fp32},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      return mlir::failure();
    }
    return srcMap.lookup(key);
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      outVals.reserve(operands[0].size());
      for (Value v : operands[0]) {
        outVals.push_back(
            cvtFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      outVals.reserve(operands[0].size());
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    size_t numElements = 4;
    if (srcElementType.isFloat8E4M3FNUZ() ||
        dstElementType.isFloat8E4M3FNUZ() ||
        srcElementType.isFloat8E5M2FNUZ() ||
        dstElementType.isFloat8E5M2FNUZ()) {
      numElements = 2;
    }
    bool useFP16IntermediateSrc =
        srcElementType.isF32() && !(isaFamily == HCU::ISAFamily::CDNA3 &&
                                    (dstElementType.isFloat8E4M3FNUZ() ||
                                     dstElementType.isFloat8E5M2FNUZ()));
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    SmallVector<Value> inVals;
    inVals.reserve(std::min(numElements, operands.size()));
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    bool isSrcFP16 = srcElementType.isF16();
    bool isSrcBF16 = srcElementType.isBF16();

    if ((isSrcFP16 || isSrcBF16) && isDstFP32) {
      SmallVector<Value> outVals;
      for (Value &v : inVals) {
        if (isSrcFP16)
          outVals.push_back(convertFp16ToFp32(loc, rewriter, v));
        else
          outVals.push_back(convertBf16ToFp32(loc, rewriter, v));
      }
      return outVals;
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = cvtFp32ToFp16(loc, rewriter, v,
                          roundingMode.value_or(RoundingMode::RTNE));
    inVals.resize(numElements, undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals;
    if (srcType != dstType) {
      auto getCvtFunc = getConversionFunc(srcType, dstType, roundingMode);
      if (failed(getCvtFunc)) {
        mlir::emitError(loc, "Unsupported conversion from ")
            << srcType << " to " << dstType
            << (roundingMode.has_value()
                    ? " with rounding mode " +
                          stringifyRoundingMode(roundingMode.value())
                    : "");
        return outVals;
      } else {
        auto cvtFunc = getCvtFunc.value();
        outVals = cvtFunc(loc, rewriter, inVals);
      }
    } else {
      outVals = inVals;
    }

    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  HCU::ISAFamily isaFamily;
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  return convertFp32ToBf16(loc, rewriter, result, RoundingMode::RTNE);
}

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

static SmallVector<Value> S8_to_Bf16(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value i32Val = sext(i32_ty, inVal);

    GCNBuilder builder;
    auto &cvt = *builder.create("v_cvt_f32_i32");
    auto res = builder.newOperand("=v");
    auto operand = builder.newOperand(i32Val, "v");
    cvt(res, operand);
    auto f32Val = builder.launch(rewriter, loc, f32_ty, false);

    f32Val = bitcast(f32Val, i32_ty);
    auto shifted = lshr(i32_ty, f32Val, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    outValues.push_back(bitcast(truncated, bf16_ty));
  }
  return outValues;
}

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    } else {
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {
          convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __ocml_exp_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0][0], f32_val(log2e));

    // Here we use llvm.exp2.f32 instead of math::Exp2Op. The latter
    // flushes denorms by default, but we want to preserve denorms by default
    // for expOp.
    StringRef funcName = "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {rewriter.create<LLVM::CallOp>(loc, funcOp, prod).getResult()};
  }
};

struct Exp2OpConversion
    : ElementwiseOpConversionBase<mlir::math::Exp2Op, Exp2OpConversion> {
  using ElementwiseOpConversionBase<
      mlir::math::Exp2Op, Exp2OpConversion>::ElementwiseOpConversionBase;

  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(mlir::math::Exp2Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __ocml_exp2_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    // On HCU backend, both intrinsics are lowered to v_exp_f32 instruction,
    // which flushes input and output denorms. `llvm.amdgcn.exp2.f32` provides
    // direct access to v_exp_f32. For `llvm.exp2.f32`, the LLVM backend inserts
    // instructions to handle denorms iff `allow_flush_denorm` is False.
    StringRef funcName = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  bool ftz;
};

} // namespace

namespace mlir::triton::HCU {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  // fmin (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MinimumFOp, LLVM::MinimumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  // fmax (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MaximumFOp, LLVM::MaximumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ElementwiseOpConversion<triton::PreciseSqrtOp, LLVM::SqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);

  // ExpOpConversionApprox will try using __ocml_exp2_f32 if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // later pass will call __ocml_exp_f64 for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  // Exp2OpConversion will use llvm.exp2.f32 or llvm.amdgcn.exp2.f32
  // based on the ftz flag if the input type is FP32. For FP64 input,
  // Exp2OpConversion will return failure and later pass will call
  // __ocml_exp2_f64 for higher-precision calculation
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis,
      /*hwNanPropagationSupported=*/false, benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}
} // namespace mlir::triton::HCU
