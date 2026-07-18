#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir::triton::gpu;

namespace mlir::triton {

namespace gpu {
namespace {

//===----------------------------------------------------------------------===//
// Data type conversion utility functions
//===----------------------------------------------------------------------===//
template <typename FPType> struct FPTypeInfo {
  FPTypeInfo(Location loc, ConversionPatternRewriter &rewriter,
             TritonLLVMOpBuilder &builder)
      : loc(loc), rewriter(rewriter), b(builder) {}
  constexpr IntegerType getIntType() {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return i32_ty;
    }
    if constexpr (std::is_same_v<FPType, Float16Type> ||
                  std::is_same_v<FPType, BFloat16Type>) {
      return i16_ty;
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType> ||
                  std::is_same_v<FPType, Float8E5M2Type> ||
                  std::is_same_v<FPType, Float8E4M3FNUZType> ||
                  std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return i8_ty;
    }
    return nullptr;
  }

  auto getHalfwayPointsForDstType(TypeID dstTyID) {
    using VecType =
        std::conditional_t<std::is_same_v<FPType, Float32Type>,
                           SmallVector<int32_t>, SmallVector<int16_t>>;
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x3a800000,  // halfway between [0/8 * 2^-6, 1/8 * 2^-6]
                       0x3b400000,  // halfway between [1/8 * 2^-6, 2/8 * 2^-6]
                       0x3ba00000,  // halfway between [2/8 * 2^-6, 3/8 * 2^-6]
                       0x3be00000,  // halfway between [3/8 * 2^-6, 4/8 * 2^-6]
                       0x3c100000,  // halfway between [4/8 * 2^-6, 5/8 * 2^-6]
                       0x3c300000,  // halfway between [5/8 * 2^-6, 6/8 * 2^-6]
                       0x3c500000,  // halfway between [6/8 * 2^-6, 7/8 * 2^-6]
                       0x3c700000}; // halfway between [7/8 * 2^-6, 8/8 * 2^-6]
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{
            0x37000000,  // halfway between [0/4 * 2^(-14), 1/4 * 2^(-14)]
            0x37c00000,  // halfway between [1/4 * 2^(-14), 2/4 * 2^(-14)]
            0x38200000,  // halfway between [2/4 * 2^(-14), 3/4 * 2^(-14)]
            0x38600000}; // halfway between [3/4 * 2^(-14), 4/4 * 2^(-14)]
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float32)
        return VecType{0x3a000000,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x3ac00000,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x3b200000,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x3b600000,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x3b900000,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x3bb00000,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x3bd00000,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x3bf00000}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
      if (dstTyID == TypeID::get<Float8E5M2FNUZType>())
        // Minimum normal for E5M2FNUZ is 0x38000000 (2^-15)
        // We divide the range of subnormals in 2^2 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float32)
        return VecType{
            0x36800000,  // halfway between [0/4 * 2^-15, 1/4 * 2^-15]
            0x37400000,  // halfway between [1/4 * 2^-15, 2/4 * 2^-15]
            0x37a00000,  // halfway between [2/4 * 2^-15, 3/4 * 2^-15]
            0x37e00000}; // halfway between [3/4 * 2^-15, 4/4 * 2^-15]
    }
    if constexpr (std::is_same_v<FPType, Float16Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x1400, 0x1A00, 0x1D00, 0x1F00,
                       0x2080, 0x2180, 0x2280, 0x2380};
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{0x0080, 0x0180, 0x0200, 0x0380};
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // Minimum normal for E4M3FNUZ is 0x2000 (2^-7)
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here float16)
        return VecType{0x1000,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x1600,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x1900,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x1b00,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x1c80,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x1d80,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x1e80,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x1f80}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
    }
    if constexpr (std::is_same_v<FPType, BFloat16Type>) {
      if (dstTyID == TypeID::get<Float8E4M3FNUZType>())
        // Minimum normal for E4M3FNUZ is 0x3c00 (2^-7)
        // We divide the range of subnormals in 2^3 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here bfloat16)
        return VecType{0x3a00,  // halfway between [0/8 * 2^-7, 1/8 * 2^-7]
                       0x3ac0,  // halfway between [1/8 * 2^-7, 2/8 * 2^-7]
                       0x3b20,  // halfway between [2/8 * 2^-7, 3/8 * 2^-7]
                       0x3b60,  // halfway between [3/8 * 2^-7, 4/8 * 2^-7]
                       0x3b90,  // halfway between [4/8 * 2^-7, 5/8 * 2^-7]
                       0x3bb0,  // halfway between [5/8 * 2^-7, 6/8 * 2^-7]
                       0x3bd0,  // halfway between [6/8 * 2^-7, 7/8 * 2^-7]
                       0x3bf0}; // halfway between [7/8 * 2^-7, 8/8 * 2^-7]
      if (dstTyID == TypeID::get<Float8E5M2FNUZType>()) {
        // Minimum normal for E5M2FNUZ is 0x3800 (2^-15)
        // We divide the range of subnormals in 2^2 subranges.
        // Each i entry in the LUT corresponds to the midpoint of the ith
        // subrange represented in the src format (here bfloat16)
        // 2^-18 =
        return VecType{0x3680,  // halfway between [0/4 * 2^-15, 1/4 * 2^-15]
                       0x3740,  // halfway between [1/4 * 2^-15, 2/4 * 2^-15]
                       0x37a0,  // halfway between [2/4 * 2^-15, 3/4 * 2^-15]
                       0x37e0}; // halfway between [3/4 * 2^-15, 4/4 * 2^-15]
      }
      if (dstTyID == TypeID::get<Float8E4M3FNType>())
        return VecType{0x3a80, 0x3b40, 0x3ba0, 0x3be0,
                       0x3c10, 0x3c30, 0x3c50, 0x3c70};
      if (dstTyID == TypeID::get<Float8E5M2Type>())
        return VecType{0x3700, 0x37c0, 0x3820, 0x3860};
    }
    return VecType{};
  }

  constexpr Value toLLVMIntValue(int32_t val) {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return b.i32_val(val);
    }
    if constexpr (std::is_same_v<FPType, Float16Type> ||
                  std::is_same_v<FPType, BFloat16Type>) {
      return b.i16_val(val);
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType> ||
                  std::is_same_v<FPType, Float8E5M2Type> ||
                  std::is_same_v<FPType, Float8E4M3FNUZType> ||
                  std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return b.i8_val(val);
    }
    return nullptr;
  }

  const llvm::fltSemantics &getFPSemantics() {
    if constexpr (std::is_same_v<FPType, Float32Type>) {
      return llvm::APFloat::IEEEsingle();
    }
    if constexpr (std::is_same_v<FPType, Float16Type>) {
      return llvm::APFloat::IEEEhalf();
    }
    if constexpr (std::is_same_v<FPType, BFloat16Type>) {
      return llvm::APFloat::BFloat();
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNType>) {
      return llvm::APFloat::Float8E4M3FN();
    }
    if constexpr (std::is_same_v<FPType, Float8E4M3FNUZType>) {
      return llvm::APFloat::Float8E4M3FNUZ();
    }
    if constexpr (std::is_same_v<FPType, Float8E5M2FNUZType>) {
      return llvm::APFloat::Float8E5M2FNUZ();
    }

    return llvm::APFloat::Bogus();
  }

  Location loc;
  ConversionPatternRewriter &rewriter;
  TritonLLVMOpBuilder &b;
};

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &,
                                         triton::FpToFpOp)>
    ConverterT;

static Value Fp8E4M3FN_to_Fp32_oneValue(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a = b.undef(fp8x4VecTy);
  a = b.insert_element(fp8x4VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x4VecTy, a, b.int_val(8, 0), b.i32_val(1));
  a = b.insert_element(fp8x4VecTy, a, b.int_val(8, 0), b.i32_val(2));
  a = b.insert_element(fp8x4VecTy, a, v, b.i32_val(3));
  a = b.bitcast(a, i32_ty);

  // Get sign and absolute value
  Value sign = b.and_(a, b.int_val(32, 0x8000));
  a = b.and_(a, b.int_val(32, 0x7FFFFFFFF));

  // Right shift 1 bit to adjust the positions of exponent and mantissa
  a = b.lshr(a, b.int_val(32, 4));

  // Adjust exponent, (127 - 7) << 23 === 0xF000000
  a = b.add(a, b.int_val(32, 0xF000000));

  // Check NaN
  Value vAbs = b.and_(b.bitcast(v, i8_ty), b.int_val(8, 0x7F));
  a = b.select(b.icmp_eq(vAbs, b.int_val(8, 0x7F)), b.int_val(32, 0x7E00), a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  constexpr size_t lutSize = 8;
  static constexpr int denormsAndZeroLut[lutSize] = {
      0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};

  for (int i = 0; i < lutSize; i++) {
    a = b.select(b.icmp_eq(vAbs, b.int_val(8, i)),
                 b.int_val(32, denormsAndZeroLut[i]), a);
  }

  // Set sign
  a = b.or_(a, sign);
  a = b.bitcast(a, f32_ty);

  return a;
}

// convert fp8 to fp32
static SmallVector<Value> cvtFp8ToFp32(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v0, Value v1,
                                       const std::string &fp8_format) {
  SmallVector<Value> result(2);
  result[0] = Fp8E4M3FN_to_Fp32_oneValue(loc, rewriter, v0);
  result[1] = Fp8E4M3FN_to_Fp32_oneValue(loc, rewriter, v1);
  return result;
}

static Value Fp8E4M3FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  auto e_mask = b.int_val(16, 0x7A00);
  auto e = b.and_(i16_ty, a, e_mask);

  auto m = b.and_(i16_ty, a, b.int_val(16, 0x0700));
  auto sign = b.and_(i16_ty, a, b.int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = b.icmp_eq(e, b.int_val(16, 0x0));
  auto b0 = b.and_(i16_ty, a, b.int_val(16, 0x7FFF));
  auto b1 = b.lshr(i16_ty, b0, b.int_val(16, 1));

  // case 1, e is nonzero, add exponent by 6
  auto o0v = b.add(i16_ty, b1, b.int_val(16, 0x0C00));
  auto o0 = b.or_(i16_ty, o0v, sign);

  // case 2, e is nonzero, add exponent by 7
  auto o1v = b.add(i16_ty, b1, b.int_val(16, 0x1C00));
  auto o1 = b.or_(i16_ty, o1v, sign);

  auto io = b.select(e_is_zero, o0, o1);
  return b.bitcast(io, f16_ty);
}

static Value Fp8E4M3FN_to_Fp16_oneValue(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);
  Value v_16 = a;

  // Get sign and absolute value
  Value sign = b.and_(a, b.int_val(16, 0x8000));
  a = b.and_(a, b.int_val(16, 0x7FFF));

  Value vAbs = b.and_(v_16, b.int_val(16, 0x7FFF));

  // Compute renorm shift
  Value renorm_shift = b.int_val(16, 0);
  renorm_shift = b.select(b.icmp_ule(vAbs, b.int_val(16, 0x0700)),
                          b.int_val(16, 1), renorm_shift);
  renorm_shift = b.select(b.icmp_ule(vAbs, b.int_val(16, 0x0300)),
                          b.int_val(16, 2), renorm_shift);
  renorm_shift = b.select(b.icmp_eq(vAbs, b.int_val(16, 0x0100)),
                          b.int_val(16, 3), renorm_shift);

  // Right shift 1 bit to adjust the positions of exponent and mantissa
  a = b.shl(a, renorm_shift);
  a = b.lshr(a, b.int_val(16, 1));

  // Adjust exponent compensator, (15 - 7 - renorm_shift) << 10
  Value exponent_compensator =
      b.shl(b.sub(b.int_val(16, 8), renorm_shift), b.int_val(16, 10));
  a = b.add(a, exponent_compensator);

  // Check NaN
  a = b.select(b.icmp_eq(vAbs, b.int_val(16, 0x7F00)), b.int_val(16, 0x7E00),
               a);

  // Check Zero
  a = b.select(b.icmp_eq(vAbs, b.int_val(16, 0x0000)), b.int_val(16, 0x0000),
               a);

  // Check denorms and zero
  // Here we use a LUT to map S.0000.000 ~ S.0000.111 to its corresponding fp16
  // value
  // constexpr size_t lutSize = 8;
  // static constexpr int denormsAndZeroLut[lutSize] = {
  //     0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};

  // Set sign
  a = b.or_(a, sign);
  a = b.bitcast(a, f16_ty);

  return a;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v, triton::FpToFpOp op) {
  SmallVector<Value> result(2);
  result[0] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E4M3FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E4M3FN_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &values, triton::FpToFpOp op) {
  SmallVector<Value> results(2);
  results[0] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[0]);
  results[1] = Fp8E4M3FN_to_Fp16_oneValue(loc, rewriter, values[1]);
  return results;
}

static SmallVector<Value>
Fp8E4M3FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v, triton::FpToFpOp op) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
}

static SmallVector<Value> Fp8E4M3FN_to_Fp32(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v,
                                            triton::FpToFpOp op) {
  assert(v.size() == 2);
  return cvtFp8ToFp32(loc, rewriter, v[0], v[1], "fp8");
}

static SmallVector<Value>
Fp8E5M2_to_Fp16_SW(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v, triton::FpToFpOp op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);
  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = b.bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = b.bitcast(a1, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(1))};
}

static Value Fp8E5M2FNUZ_to_Fp16_oneValue(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value a = b.undef(fp8x2VecTy);
  a = b.insert_element(fp8x2VecTy, a, b.int_val(8, 0), b.i32_val(0));
  a = b.insert_element(fp8x2VecTy, a, v, b.i32_val(1));
  a = b.bitcast(a, i16_ty);

  auto e = b.and_(i16_ty, a, b.int_val(16, 0x7C00));
  auto m = b.and_(i16_ty, a, b.int_val(16, 0x0300));
  auto sign = b.and_(i16_ty, a, b.int_val(16, 0x8000));

  // check whether all exponents are zeros
  auto e_is_zero = b.icmp_eq(e, b.int_val(16, 0x0));

  // case 1, e is zero, need to move m right by 1 bit
  auto m1 = b.lshr(i16_ty, m, b.int_val(16, 1));
  auto o0 = b.or_(i16_ty, sign, m1);

  // case 2, e is nonzero, sub exponent by 1
  auto e1 = b.sub(i16_ty, e, b.int_val(16, 0x0400));

  auto e_is_one = b.icmp_eq(e, b.int_val(16, 0x0400));
  auto m2 = b.add(i16_ty, m1, b.int_val(16, 0x0200));

  auto o1 = b.or_(i16_ty, sign, b.or_(i16_ty, m, e1));
  auto o2 = b.or_(i16_ty, sign, m2);

  auto o12 = b.select(e_is_one, o2, o1);
  auto o = b.select(e_is_zero, o0, o12);

  return b.bitcast(o, f16_ty);
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v, triton::FpToFpOp op) {
  SmallVector<Value> result(2);
  result[0] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[0]);
  result[1] = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[1]);
  return result;
}

static SmallVector<Value>
Fp8E5M2FNUZ_to_Fp32(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v, triton::FpToFpOp op) {
  SmallVector<Value> result(2);
  for (int i = 0; i < 2; i++) {
    Value fp16_val = Fp8E5M2FNUZ_to_Fp16_oneValue(loc, rewriter, v[i]);
    result[i] = rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16_val);
  }
  return result;
}

static SmallVector<Value> Fp8E5M2_to_Fp32(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v,
                                          triton::FpToFpOp op) {
  SmallVector<Value> result(4);
  SmallVector<Value> fp16_vals = Fp8E5M2_to_Fp16_SW(loc, rewriter, v, op);
  for (int i = 0; i < 4; i++) {
    result[i] = rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16_vals[i]);
  }
  return result;
}

static SmallVector<Value> Fp8E5M2_to_Bf16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v,
                                          triton::FpToFpOp op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  b0 = b.lshr(i32_ty, b0, b.i32_val(3));
  b1 = b.lshr(i32_ty, b1, b.i32_val(3));

  Value c0 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c1 = b.and_(i32_ty, b0, b.i32_val(0xFFFF0000));
  Value c2 = b.shl(i32_ty, b1, b.i32_val(16));
  Value c3 = b.and_(i32_ty, b1, b.i32_val(0xFFFF0000));

  c0 = b.bitcast(c0, f32_ty);
  c1 = b.bitcast(c1, f32_ty);
  c2 = b.bitcast(c2, f32_ty);
  c3 = b.bitcast(c3, f32_ty);

  Value d0 = b.fmul(f32_ty, c0, b.f32_val(0x1p+112));
  Value d1 = b.fmul(f32_ty, c1, b.f32_val(0x1p+112));
  Value d2 = b.fmul(f32_ty, c2, b.f32_val(0x1p+112));
  Value d3 = b.fmul(f32_ty, c3, b.f32_val(0x1p+112));

  d0 = b.bitcast(d0, i32_ty);
  d1 = b.bitcast(d1, i32_ty);
  d2 = b.bitcast(d2, i32_ty);
  d3 = b.bitcast(d3, i32_ty);

  Value out0 = b.or_(i32_ty, b.lshr(i32_ty, d0, b.i32_val(16)), d1);
  Value out1 = b.or_(i32_ty, b.lshr(i32_ty, d2, b.i32_val(16)), d3);

  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, a1, b.i32_val(0x80008000));

  out0 = b.or_(i32_ty, out0, sign0);
  out1 = b.or_(i32_ty, out1, sign1);

  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  out0 = b.bitcast(out0, bf16x2VecTy);
  out1 = b.bitcast(out1, bf16x2VecTy);

  return {b.extract_element(i16_ty, out0, b.i32_val(0)),
          b.extract_element(i16_ty, out0, b.i32_val(1)),
          b.extract_element(i16_ty, out1, b.i32_val(0)),
          b.extract_element(i16_ty, out1, b.i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp8E5M2(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = b.undef(bf16x2VecTy);
  Value bf16x2Vec1 = b.undef(bf16x2VecTy);
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[0], b.i32_val(0));
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[1], b.i32_val(1));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[2], b.i32_val(0));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[3], b.i32_val(1));
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = b.undef(fp8x4VecTy);
  sign0 = b.bitcast(sign0, fp8x4VecTy);
  sign1 = b.bitcast(sign1, fp8x4VecTy);
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(1)),
                          b.i32_val(0));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(3)),
                          b.i32_val(1));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(1)),
                          b.i32_val(2));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(3)),
                          b.i32_val(3));
  sign = b.bitcast(sign, i32_ty);

  Value nosign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x7fff7fff));
  Value nosign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x7fff7fff));

  Value nosign_0_0 = b.and_(i32_ty, nosign0, b.i32_val(0xffff0000));
  nosign_0_0 = b.umax(i32_ty, nosign_0_0, b.i32_val(0x38000000));
  nosign_0_0 = b.umin(i32_ty, nosign_0_0, b.i32_val(0x57e00000));
  Value nosign_0_1 = b.and_(i32_ty, nosign0, b.i32_val(0x0000ffff));
  nosign_0_1 = b.umax(i32_ty, nosign_0_1, b.i32_val(0x3800));
  nosign_0_1 = b.umin(i32_ty, nosign_0_1, b.i32_val(0x57e0));
  nosign0 = b.or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = b.and_(i32_ty, nosign1, b.i32_val(0xffff0000));
  nosign_1_0 = b.umax(i32_ty, nosign_1_0, b.i32_val(0x38000000));
  nosign_1_0 = b.umin(i32_ty, nosign_1_0, b.i32_val(0x57e00000));
  Value nosign_1_1 = b.and_(i32_ty, nosign1, b.i32_val(0x0000ffff));
  nosign_1_1 = b.umax(i32_ty, nosign_1_1, b.i32_val(0x3800));
  nosign_1_1 = b.umin(i32_ty, nosign_1_1, b.i32_val(0x57e0));
  nosign1 = b.or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = b.add(i32_ty, nosign0, b.i32_val(0x00100010));
  nosign1 = b.add(i32_ty, nosign1, b.i32_val(0x00100010));
  nosign0 = b.sub(i32_ty, nosign0, b.i32_val(0x38003800));
  nosign1 = b.sub(i32_ty, nosign1, b.i32_val(0x38003800));
  nosign0 = b.shl(i32_ty, nosign0, b.i32_val(3));
  nosign1 = b.shl(i32_ty, nosign1, b.i32_val(3));

  nosign0 = b.bitcast(nosign0, fp8x4VecTy);
  nosign1 = b.bitcast(nosign1, fp8x4VecTy);
  Value nosign = b.undef(fp8x4VecTy);
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(1)),
                            b.i32_val(0));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(3)),
                            b.i32_val(1));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(1)),
                            b.i32_val(2));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(3)),
                            b.i32_val(3));
  nosign = b.bitcast(nosign, i32_ty);

  Value fp8x4Vec = b.or_(i32_ty, nosign, sign);
  fp8x4Vec = b.bitcast(fp8x4Vec, fp8x4VecTy);
  return {b.extract_element(i8_ty, fp8x4Vec, b.i32_val(0)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(1)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(2)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(3))};
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

  explicit ElementwiseOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase<SourceOp, ElementwiseOpConversion>(
            typeConverter, benefit) {}

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
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<LLVM::FPExtOp>(loc, f32_ty, v);
  }

  template <typename T>
  static Value
  convertFp32ToBf16(Location loc, T op, ConversionPatternRewriter &rewriter,
                    const Value &v,
                    const RoundingMode rounding = RoundingMode::RTNE) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (rounding == RoundingMode::RTZ) {
      auto as_int32 = b.bitcast(v, i32_ty);
      auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
      auto truncated = b.trunc(i16_ty, shifted);
      return b.bitcast(truncated, i16_ty);
    } else if (rounding == RoundingMode::RTNE_NO_NAN) {
      // This implementation method considers the case of low bit carry,
      // so the precsion is higher.
      // This method will get wrong result when input have NAN,
      auto as_int32 = b.bitcast(v, i32_ty);
      auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
      auto and_one = b.and_(i32_ty, shifted, b.i32_val(1));
      auto add_32767 = b.add(i32_ty, as_int32, b.i32_val(32767));
      auto add_value = b.add(i32_ty, add_32767, and_one);
      auto shifted_add = b.lshr(i32_ty, add_value, b.i32_val(16));
      auto truncated = b.trunc(i16_ty, shifted_add);
      return (b.bitcast(truncated, i16_ty));
    } else {
      // use the __float2bfloat16 method instead
      // default way: rounding == RoundingMode::RTNE
      std::string instructionName = "llvm.mxc.cvt.f32tobf16";
      StringRef funcName(instructionName);
      ValueRange valueRange({v});
      Type funcType = mlir::triton::gpu::getFunctionType(i16_ty, valueRange);
      LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, funcName, funcType);
      auto ret =
          LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange).getResult();
      return ret;
    }
  }

  static SmallVector<Value>
  Fp32_to_Fp8E5M2_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v, triton::FpToFpOp op) {
    assert(v.size() == 2);
    // auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> result(2);
    for (size_t i = 0; i < 2; ++i) {
      Value fp32 = v[i];
      Value i32 = b.bitcast(fp32, i32_ty);

      Value s = b.and_(i32_ty, i32, b.i32_val(0x80000000));
      Value exp =
          b.and_(i32_ty, b.lshr(i32_ty, i32, b.i32_val(23)), b.i32_val(0xFF));
      Value man = b.and_(i32_ty, i32, b.i32_val(0x007FFFFF));

      // Convert 8-bit exponent to 5-bit
      Value exp5 = b.select(b.icmp_ult(exp, b.i32_val(0x71)), b.i32_val(0),
                            b.sub(i32_ty, exp, b.i32_val(0x70)));

      // Handle subnormal values (exp5 = 0)
      // - exp <  0x6e: mantissa = 0x00000000 (0)
      // - exp == 0x6e: mantissa = 0x00000000 (0),
      //                           0x00200000 (1/4)
      // - exp == 0x6f: mantissa = 0x00200000 (1/4),
      //                           0x00400000 (1/2)
      // - exp == 0x70: mantissa = 0x00400000 (1/2),
      //                           0x00600000 (3/4),
      //                           0x00800000 (1)
      man = b.select(b.icmp_ult(exp, b.i32_val(0x6e)), b.i32_val(0), man);
      man = b.select(b.icmp_eq(exp, b.i32_val(0x6e)),
                     b.select(b.icmp_ne(man, b.i32_val(0)),
                              b.i32_val(0x00200000), b.i32_val(0)),
                     man);
      man = b.select(b.icmp_eq(exp, b.i32_val(0x6f)),
                     b.select(b.icmp_uge(man, b.i32_val(0x00400000)),
                              b.i32_val(0x00400000), b.i32_val(0x00200000)),
                     man);
      man = b.select(
          b.icmp_eq(exp, b.i32_val(0x70)),
          b.select(b.icmp_ugt(man, b.i32_val(0x00200000)),
                   b.select(b.icmp_uge(man, b.i32_val(0x00600000)),
                            b.i32_val(0x00800000), b.i32_val(0x00600000)),
                   b.i32_val(0x00400000)),
          man);

      // Round 23-bit mantissa to 2-bit nearest, ties to even
      Value sig = b.or_(i32_ty, b.shl(i32_ty, exp5, b.i32_val(23)), man);
      Value bias =
          b.add(i32_ty,
                b.lshr(i32_ty, b.and_(i32_ty, sig, b.i32_val(0x00200000)),
                       b.i32_val(21)),
                b.i32_val(0x000FFFFF));
      i32 = b.add(i32_ty, sig, bias);

      // Handle overflow using saturation mode, by setting sig to be the max.
      // Overflow will happe for the following cases:
      // - Any number equal or larger than 0x0F700000 after rounding
      // - Exponent larged than 0x8E (including infinite 0xFF)
      i32 = b.select(b.or_(b.icmp_ugt(exp, b.i32_val(0x8E)),
                           b.icmp_uge(sig, b.i32_val(0x0F700000))),
                     b.i32_val(0x0F7FFFFF), i32);

      // Handle NaN value by keeping it Nan
      i32 = b.select(b.and_(b.icmp_eq(exp, b.i32_val(0xFF)),
                            b.icmp_ne(man, b.i32_val(0x0))),
                     b.i32_val(0x0FC00000), i32);

      // Add sign bit
      i32 = b.or_(i32_ty, b.lshr(i32_ty, s, b.i32_val(3)), i32);

      // Truncate to 8-bit
      result[i] = b.trunc(i8_ty, b.lshr(i32_ty, i32, b.i32_val(21)));
    }
    return result;
  }

  template <typename SrcFPType, typename DstFPType>
  static Value downcastToFp8_RTNE_oneValue(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Value v) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    static_assert((std::is_same_v<SrcFPType, Float32Type>) ||
                  (std::is_same_v<SrcFPType, Float16Type>) ||
                  (std::is_same_v<SrcFPType, BFloat16Type>));
    static_assert((std::is_same_v<DstFPType, Float8E4M3FNType> ||
                   std::is_same_v<DstFPType, Float8E4M3FNUZType> ||
                   std::is_same_v<DstFPType, Float8E5M2FNUZType>));
    constexpr bool isFp8UZ = (std::is_same_v<DstFPType, Float8E4M3FNUZType> ||
                              std::is_same_v<DstFPType, Float8E5M2FNUZType>);

    FPTypeInfo<SrcFPType> srcFpInfo(loc, rewriter, b);
    FPTypeInfo<DstFPType> dstFpInfo(loc, rewriter, b);

    const llvm::fltSemantics &srcSemantic = srcFpInfo.getFPSemantics();
    auto srcWidth = llvm::APFloat::getSizeInBits(srcSemantic);
    auto srcMantissaBits = llvm::APFloat::semanticsPrecision(srcSemantic) - 1;
    auto srcExponentBits = srcWidth - srcMantissaBits - 1;
    auto srcBias = (1 << (srcExponentBits - 1)) - 1;

    const llvm::fltSemantics &dstSemantic = dstFpInfo.getFPSemantics();
    auto dstWidth = llvm::APFloat::getSizeInBits(dstSemantic);
    auto dstMantissaBits = llvm::APFloat::semanticsPrecision(dstSemantic) - 1;
    auto dstExponentBits = dstWidth - dstMantissaBits - 1;
    auto dstBias = (1 << (dstExponentBits - 1)) - 1;
    if (isFp8UZ) {
      dstBias++;
    }

    auto srcIntType = srcFpInfo.getIntType();
    // Value isNaN = checkIsNan(b, v);

    uint32_t reducedMantissaBits = srcMantissaBits - dstMantissaBits;
    Value reducedMantissaValue = srcFpInfo.toLLVMIntValue(reducedMantissaBits);

    // Get sign and absolute value
    Value intVal = b.bitcast(v, srcIntType);
    uint32_t signMask = 1 << (srcWidth - 1);
    Value sign = b.trunc(
        i8_ty, b.lshr(b.and_(intVal, srcFpInfo.toLLVMIntValue(signMask)),
                      srcFpInfo.toLLVMIntValue(srcWidth - 8)));

    uint32_t absoluteMask = signMask - 1;
    intVal = b.and_(intVal, srcFpInfo.toLLVMIntValue(absoluteMask));

    uint32_t expFullMask = ((1U << srcExponentBits) - 1U) << srcMantissaBits;
    uint32_t mantFullMask = 1U << srcMantissaBits - 1U;
    Value isNaNOrInf =
        b.icmp_eq(b.and_(intVal, srcFpInfo.toLLVMIntValue(expFullMask)),
                  srcFpInfo.toLLVMIntValue(expFullMask));
    Value notInf =
        b.icmp_ne(b.and_(intVal, srcFpInfo.toLLVMIntValue(mantFullMask)),
                  srcFpInfo.toLLVMIntValue(0));
    Value isNaN = b.and_(isNaNOrInf, notInf);

    // Rounding to nearest even
    uint32_t baseRoundingBias = (1 << (reducedMantissaBits - 1)) - 1;

    // For Fp16, S.EEEEE.MMMMMMMMMM => 0.00000.00M0000000 => 0.00000.000000000M
    uint32_t mantissaLSB = 1 << reducedMantissaBits;
    Value mantissaLSBValue = srcFpInfo.toLLVMIntValue(mantissaLSB);
    Value remainingMantissaLSB =
        b.lshr(b.and_(intVal, mantissaLSBValue), reducedMantissaValue);
    Value roundingBias =
        b.add(remainingMantissaLSB, srcFpInfo.toLLVMIntValue(baseRoundingBias));
    Value vFp8 = b.add(intVal, roundingBias);

    // Reduce mantissa to number of bits of the destination format
    // Example: For Fp16 to FP8E4M3FN, reduceMantissaMask == 1.11111.1110000000
    uint32_t reduceMantissaMask =
        ((1 << (1 + srcExponentBits + dstMantissaBits + 1)) - 1)
        << reducedMantissaBits;
    Value reduceMantissa = srcFpInfo.toLLVMIntValue(reduceMantissaMask);
    vFp8 = b.and_(vFp8, reduceMantissa);

    // We round numbers smaller than the minimal normal number in Fp8 to make
    // it easier to handle subnormals
    auto dstSmallest = llvm::APFloat::getSmallestNormalized(dstSemantic);
    // Get the srcFpType representation of the minimal normal number in Fp8
    bool losesInfo;
    dstSmallest.convert(srcSemantic, APFloat::rmNearestTiesToEven, &losesInfo);
    uint32_t dstMinimal =
        static_cast<uint32_t>(dstSmallest.bitcastToAPInt().getZExtValue());
    vFp8 = b.umax(vFp8, srcFpInfo.toLLVMIntValue(dstMinimal));

    // Adjust exponent bias
    uint32_t expBias = (srcBias - dstBias) << srcMantissaBits;
    vFp8 = b.sub(vFp8, srcFpInfo.toLLVMIntValue(expBias));

    // Shift right and truncate
    vFp8 = b.trunc(i8_ty, b.lshr(vFp8, reducedMantissaValue));

    // Any numbers larger than the max normal number(including infinity) in FP8
    // after rounding will cause overflow
    auto dstLargest = llvm::APFloat::getLargest(dstSemantic);
    uint32_t dstMaxPositive =
        static_cast<uint32_t>(dstLargest.bitcastToAPInt().getZExtValue());
    // Get the srcFpType representation of the maximal normal number in Fp8
    dstLargest.convert(srcSemantic, APFloat::rmNearestTiesToEven, &losesInfo);
    uint32_t dstMaxOfSrcType =
        static_cast<uint32_t>(dstLargest.bitcastToAPInt().getZExtValue());

    // For Fp16, 0x5F7F == 0.10111.1101111111 is the largest possible normal
    // number(including infinity) after rounding in FP8E4M3
    // For Fp8 UZ types, conversion with saturation converts infinity to NaN
    if constexpr (!isFp8UZ) {
      // Include infinity
      if constexpr (std::is_same_v<SrcFPType, Float32Type>)
        dstMaxOfSrcType |= 0x7ffff;
      else if constexpr (std::is_same_v<SrcFPType, Float16Type>)
        dstMaxOfSrcType |= 0x7f;
      else
        dstMaxOfSrcType |= 0x7;
    } else {
      uint32_t expFullMask = ((1U << srcExponentBits) - 1U) << srcMantissaBits;
      // In case the exponent is full (all ones), then we have either a NaN or
      // Inf
      Value isNaNOrInf =
          b.icmp_eq(b.and_(intVal, srcFpInfo.toLLVMIntValue(expFullMask)),
                    srcFpInfo.toLLVMIntValue(expFullMask));
      isNaN = isNaNOrInf;
    }

    Value isOverflow =
        b.icmp_ugt(intVal, srcFpInfo.toLLVMIntValue(dstMaxOfSrcType));
    vFp8 = b.select(isOverflow, dstFpInfo.toLLVMIntValue(dstMaxPositive), vFp8);

    // Round subnormals to nearest even. Ref:
    // https://githucom/openxla/xla/blob/f20c6fe2/xla/service/elemental_ir_emitter.cc#L272
    auto dstTyID = TypeID::get<DstFPType>();
    auto halfwayPointsLUT = srcFpInfo.getHalfwayPointsForDstType(dstTyID);
    size_t lutSize = halfwayPointsLUT.size();

    for (int i = lutSize - 1; i >= 0; i--) {
      Value cmp;
      if (i % 2 == 0) {
        cmp = b.icmp_ule(intVal, srcFpInfo.toLLVMIntValue(halfwayPointsLUT[i]));
      } else {
        cmp = b.icmp_ult(intVal, srcFpInfo.toLLVMIntValue(halfwayPointsLUT[i]));
      }

      vFp8 = b.select(cmp, b.i8_val(i), vFp8);
    }

    int32_t positiveNan = 0;
    if constexpr (isFp8UZ) {
      // Only one NaN value which is represented with sign = 1
      positiveNan = (1 << (dstExponentBits + dstMantissaBits));
    } else {
      positiveNan = (1 << (dstExponentBits + dstMantissaBits)) - 1;
    }

    // NaN remains NaN after conversion
    vFp8 = b.select(isNaN, dstFpInfo.toLLVMIntValue(positiveNan), vFp8);

    // Set sign bit
    vFp8 = b.or_(vFp8, sign);
    // In UZ formats there is only 1 zero (positive zero)
    // Correct negative zero to 0
    if constexpr (isFp8UZ) {
      Value isNegativeZero = b.and_(b.icmp_eq(vFp8, b.i8_val(0x80)),
                                    b.icmp_eq(isNaN, b.i1_val(0)));
      vFp8 = b.select(isNegativeZero, b.i8_val(0), vFp8);
    }

    return vFp8;
  }

  static Value
  convertFp32DowncastCommon(Location loc, ConversionPatternRewriter &rewriter,
                            const Value &v, const RoundingMode rounding,
                            const int &exp_bits, const int &mantissa_bits,
                            const int &exp_bias) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value as_int32 = b.bitcast(v, i32_ty);
    int exp_bias_fp32 = 127;
    int exp_shift = exp_bias_fp32 - exp_bias;
    // extract sign, exp and mantissa
    Value sign = b.and_(as_int32, b.i32_val(0x80000000));
    Value mantissa = b.and_(as_int32, b.i32_val(0x007fffff));
    Value exp = b.and_(b.lshr(as_int32, b.i32_val(23)), b.i32_val(0xff));

    mantissa = b.select(b.icmp_eq(exp, b.int_val(32, 0)), mantissa,
                        b.add(mantissa, b.int_val(32, 0x800000)));
    exp = b.select(b.icmp_eq(exp, b.int_val(32, 0)), exp,
                   b.sub(exp, b.int_val(32, 1)));

    if (rounding == RoundingMode::RTNE) {
      int shift_bits = 23 - mantissa_bits;
      int rounding_bias = (1 << (shift_bits - 1)) - 1;
      mantissa = b.add(i32_ty, mantissa, b.i32_val(rounding_bias));
    }

    // process denormal cases
    Value new_mantissa = b.select(
        b.icmp_ule(exp, b.int_val(32, exp_shift - 16)), b.int_val(32, 0),
        b.lshr(mantissa, b.int_val(32, 23 - mantissa_bits)));
    Value new_exp = b.select(b.icmp_ule(exp, b.int_val(32, exp_shift - 16)),
                             b.int_val(32, exp_shift),
                             exp); // too small for fp16 to express
    new_mantissa =
        b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 8)),
                 new_mantissa, b.lshr(new_mantissa, b.int_val(32, 8)));
    new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 8)),
                       new_exp, b.add(new_exp, b.int_val(32, 8)));
    new_mantissa =
        b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 4)),
                 new_mantissa, b.lshr(new_mantissa, b.int_val(32, 4)));
    new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 4)),
                       new_exp, b.add(new_exp, b.int_val(32, 4)));
    new_mantissa =
        b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 2)),
                 new_mantissa, b.lshr(new_mantissa, b.int_val(32, 2)));
    new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 2)),
                       new_exp, b.add(new_exp, b.int_val(32, 2)));
    new_mantissa =
        b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 1)),
                 new_mantissa, b.lshr(new_mantissa, b.int_val(32, 1)));
    new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 1)),
                       new_exp, b.add(new_exp, b.int_val(32, 1)));
    new_exp = b.sub(new_exp, b.i32_val(exp_shift));

    // concat sign, exp and mantissa
    int target_dtype_bits = 1 + exp_bits + mantissa_bits;
    Value new_sign = b.lshr(sign, b.i32_val(32 - target_dtype_bits));
    new_exp = b.shl(new_exp, b.i32_val(mantissa_bits));
    Value new_value =
        b.add(i32_ty, b.add(i32_ty, new_sign, new_exp), new_mantissa);
    if (target_dtype_bits == 16) {
      return b.trunc(i16_ty, new_value);
    }
    return b.trunc(i8_ty, new_value);
  }

  static Value
  convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                    const Value &v,
                    const RoundingMode rounding = RoundingMode::RTNE) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (rounding == RoundingMode::RTZ) {
      Value as_int32 = b.bitcast(v, i32_ty);
      int exp_bias_fp32 = 127;
      int exp_bias_fp16 = 15;
      int exp_shift = exp_bias_fp32 - exp_bias_fp16;
      // extract sign, exp and mantissa
      Value sign = b.and_(as_int32, b.i32_val(0x80000000));
      Value mantissa = b.and_(as_int32, b.i32_val(0x007fffff));
      Value exp = b.and_(b.lshr(as_int32, b.i32_val(23)), b.i32_val(0xff));

      mantissa = b.select(b.icmp_eq(exp, b.int_val(32, 0)), mantissa,
                          b.add(mantissa, b.int_val(32, 0x800000)));
      exp = b.select(b.icmp_eq(exp, b.int_val(32, 0)), exp,
                     b.sub(exp, b.int_val(32, 1)));

      Value new_mantissa =
          b.select(b.icmp_ule(exp, b.int_val(32, exp_shift - 16)),
                   b.int_val(32, 0), b.lshr(mantissa, b.int_val(32, 23 - 10)));
      Value new_exp = b.select(b.icmp_ule(exp, b.int_val(32, exp_shift - 16)),
                               b.int_val(32, exp_shift),
                               exp); // too small for fp16 to express
      new_mantissa =
          b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 8)),
                   new_mantissa, b.lshr(new_mantissa, b.int_val(32, 8)));
      new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 8)),
                         new_exp, b.add(new_exp, b.int_val(32, 8)));
      new_mantissa =
          b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 4)),
                   new_mantissa, b.lshr(new_mantissa, b.int_val(32, 4)));
      new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 4)),
                         new_exp, b.add(new_exp, b.int_val(32, 4)));
      new_mantissa =
          b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 2)),
                   new_mantissa, b.lshr(new_mantissa, b.int_val(32, 2)));
      new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 2)),
                         new_exp, b.add(new_exp, b.int_val(32, 2)));
      new_mantissa =
          b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 1)),
                   new_mantissa, b.lshr(new_mantissa, b.int_val(32, 1)));
      new_exp = b.select(b.icmp_ugt(new_exp, b.int_val(32, exp_shift - 1)),
                         new_exp, b.add(new_exp, b.int_val(32, 1)));
      new_exp = b.sub(new_exp, b.i32_val(exp_shift));

      Value new_sign = b.lshr(sign, b.i32_val(16));
      new_exp = b.shl(new_exp, b.i32_val(10));
      Value new_value =
          b.add(i32_ty, b.add(i32_ty, new_sign, new_exp), new_mantissa);
      Value truncated = b.trunc(i16_ty, new_value);
      return b.bitcast(truncated, f16_ty);
    }
    return rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v);
  }

  // Fp32 -> OCP Fp8 (RTNZ)
  static SmallVector<Value>
  Fp32ToFp8E5M2RTZ(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v, triton::FpToFpOp op) {
    assert(v.size() == 2);
    SmallVector<Value> ret(2);
    ret[0] = convertFp32DowncastCommon(loc, rewriter, v[0], RoundingMode::RTZ,
                                       5, 2, 15);
    ret[1] = convertFp32DowncastCommon(loc, rewriter, v[1], RoundingMode::RTZ,
                                       5, 2, 15);
    return ret;
  }

  static SmallVector<Value>
  Fp32_to_Fp8E5M2FNUZ_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &v,
                              triton::FpToFpOp op) {
    assert(v.size() == 2);
    SmallVector<Value> ret(2);
    ret[0] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E5M2FNUZType>(
        loc, rewriter, v[0]);
    ret[1] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E5M2FNUZType>(
        loc, rewriter, v[1]);
    return ret;
  }

  static SmallVector<Value>
  Fp32_to_Fp8E5M2FNUZ_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &v,
                              triton::FpToFpOp op) {
    assert(false && "Fp32_to_Fp8E5M2FNUZ_RTNE_HW unsupported yet");
    SmallVector<Value> result(2);
    return result;
  }

  static std::pair<ConverterT, size_t>
  Fp32_to_Fp8E5M2FNUZ_RTNE(int computeCapability) {
    return computeCapability >= 89
               ? std::make_pair(Fp32_to_Fp8E5M2FNUZ_RTNE_HW, 4)
               : std::make_pair(Fp32_to_Fp8E5M2FNUZ_RTNE_SW, 2);
  }

  static SmallVector<Value> PK_Fp_to_Fp_Builtin(
      Location loc, ConversionPatternRewriter &rewriter,
      const SmallVector<Value> &v, triton::FpToFpOp op, Type srcElemTy,
      Type retElemTy, Type srcInstTy, Type retInstTy, StringRef instr,
      bool isCfg = false, int pk_vec = 2, Value rounding_mode = Value{},
      Value denormal_mode = Value{}, Value src_modifiers = Value{},
      Value clamp = Value{}, Value ovfl = Value{}) {
    assert(pk_vec > 1 && "PK_Fp_to_Fp_Builtin only support when pk_vec > 1");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto tensor_size = v.size();
    auto num_vec = tensor_size / pk_vec;
    auto rem_vec = tensor_size % pk_vec;
    auto srcTy = vec_ty(srcElemTy, pk_vec);
    auto retTy = vec_ty(retElemTy, pk_vec);
    SmallVector<Value> result;
    for (unsigned i = 0; i < num_vec; i++) {
      Value src = b.undef(srcTy);
      for (unsigned p = 0; p < pk_vec; p++) {
        src = b.insert_element(src, v[i * pk_vec + p], b.i32_val(p));
      }
      if (srcTy != srcInstTy)
        src = b.bitcast(src, srcInstTy);
      Value ret;
      if (isCfg) {
        ValueRange valueRange(
            {src, rounding_mode, denormal_mode, src_modifiers, clamp, ovfl});
        Type funcType =
            mlir::triton::gpu::getFunctionType(retInstTy, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, instr, funcType);
        ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                  .getResult();
      } else {
        ValueRange valueRange({src});
        Type funcType =
            mlir::triton::gpu::getFunctionType(retInstTy, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, instr, funcType);
        ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                  .getResult();
      }
      if (retTy != retInstTy)
        ret = b.bitcast(ret, retTy);
      for (unsigned j = 0; j < pk_vec; j++) {
        result.push_back(b.extract_element(retElemTy, ret, b.i32_val(j)));
      }
    }
    if (rem_vec) {
      Value src = b.undef(srcTy);
      Value _zero;
      if (srcElemTy.isF32()) {
        _zero = b.f32_val(0);
      } else if (srcElemTy.isF16()) {
        _zero = b.f16_val(0);
      } else if (srcElemTy.isBF16()) {
        _zero = b.i16_val(0);
      } else if (srcElemTy.isUnsignedInteger() ||
                 srcElemTy.isSignlessInteger()) {
        auto bits = srcElemTy.getIntOrFloatBitWidth();
        _zero = b.int_val(bits, 0);
      } else {
        assert(false && "unsupport srcElemTy");
      }
      for (unsigned m = 0; m < pk_vec; m++) {
        if (m < rem_vec) {
          src = b.insert_element(src, v[num_vec * pk_vec + m], b.i32_val(m));
        } else {
          src = b.insert_element(src, _zero, b.i32_val(m));
        }
      }
      if (srcTy != srcInstTy)
        src = b.bitcast(src, srcInstTy);
      Value ret;
      if (isCfg) {
        ValueRange valueRange(
            {src, rounding_mode, denormal_mode, src_modifiers, clamp, ovfl});
        Type funcType =
            mlir::triton::gpu::getFunctionType(retInstTy, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, instr, funcType);
        ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                  .getResult();
      } else {
        ValueRange valueRange({src});
        Type funcType =
            mlir::triton::gpu::getFunctionType(retInstTy, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, instr, funcType);
        ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                  .getResult();
      }
      if (retTy != retInstTy)
        ret = b.bitcast(ret, retTy);
      for (unsigned n = 0; n < rem_vec; n++) {
        result.push_back(b.extract_element(retElemTy, ret, b.i32_val(n)));
      }
    }
    return result;
  }

  static SmallVector<Value>
  Fp32_to_Fp8E4M3FNUZ_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &v,
                              triton::FpToFpOp op) {
    assert(v.size() == 2);
    SmallVector<Value> ret(2);
    ret[0] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNUZType>(
        loc, rewriter, v[0]);
    ret[1] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNUZType>(
        loc, rewriter, v[1]);
    return ret;
  }

  static SmallVector<Value>
  Fp32_to_Fp8E4M3FNUZ_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &v,
                              triton::FpToFpOp op) {
    assert(false && "Fp32_to_Fp8E4M3FNUZ_RTNE_HW unsupported yet");
    SmallVector<Value> result(2);
    return result;
  }

  static std::pair<ConverterT, size_t>
  Fp32_to_Fp8E4M3FNUZ_RTNE(int computeCapability) {
    return computeCapability >= 89
               ? std::make_pair(Fp32_to_Fp8E4M3FNUZ_RTNE_HW, 4)
               : std::make_pair(Fp32_to_Fp8E4M3FNUZ_RTNE_SW, 2);
  }

  static SmallVector<Value>
  Fp32_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v, triton::FpToFpOp op) {
    SmallVector<Value> result(2);
    for (size_t i = 0; i < 2; i++)
      result[i] = downcastToFp8_RTNE_oneValue<Float32Type, Float8E4M3FNType>(
          loc, rewriter, v[i]);
    return result;
  }

  static SmallVector<Value>
  Fp32_to_Fp8E4M3FN_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk4.f32tof8.cfg");
    Value rounding_mode = b.i32_val(0); // ``_MACA_FROUND_TO_NEAREST_INT``
    Value denormal_mode = b.i32_val(3); // ``_MACA_DENORMAL_FLUSH_NONE``
    Value src_modifiers = b.i32_val(0); // ``_MACA_SRC_NO_MODIFIER``
    Value clamp = b.i1_val(0);
    Value ovfl = b.i1_val(0);
    int pk_vec = 4;
    Type srcInstrTy = vec_ty(f32_ty, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, f32_ty, i8_ty, srcInstrTy,
                               i32_ty, funcName, true, pk_vec, rounding_mode,
                               denormal_mode, src_modifiers, clamp, ovfl);
  }

  static std::pair<ConverterT, size_t>
  Fp32_to_Fp8E4M3FN_RTNE(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F32_F8_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp32_to_Fp8E4M3FN_RTNE_HW, 4)
               : std::make_pair(Fp32_to_Fp8E4M3FN_RTNE_SW, 2);
  }

  static SmallVector<Value>
  Fp32_to_Fp8E5M2_RTNE_HW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk4.f32tobf8.cfg");
    Value rounding_mode = b.i32_val(0); // ``_MACA_FROUND_TO_NEAREST_INT``
    Value denormal_mode = b.i32_val(3); // ``_MACA_DENORMAL_FLUSH_NONE``
    Value src_modifiers = b.i32_val(0); // ``_MACA_SRC_NO_MODIFIER``
    Value clamp = b.i1_val(0);
    Value ovfl = b.i1_val(0);
    int pk_vec = 4;
    Type srcInstrTy = vec_ty(f32_ty, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, f32_ty, i8_ty, srcInstrTy,
                               i32_ty, funcName, true, pk_vec, rounding_mode,
                               denormal_mode, src_modifiers, clamp, ovfl);
  }

  static SmallVector<Value>
  Fp8E4M3FN_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk.f8tof16");
    Type retElemTy = f16_ty;
    int pk_vec = 2;
    Type dstInstrTy = vec_ty(retElemTy, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, i8_ty, retElemTy, i16_ty,
                               dstInstrTy, funcName);
  }

  static SmallVector<Value>
  Fp8E5M2_to_Fp16_HW(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk.bf8tof16");
    Type retElemTy = f16_ty;
    int pk_vec = 2;
    Type dstInstrTy = vec_ty(retElemTy, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, i8_ty, retElemTy, i16_ty,
                               dstInstrTy, funcName);
  }

  static SmallVector<Value>
  Fp8E4M3FN_to_Bf16_FAST(Location loc, ConversionPatternRewriter &rewriter,
                         const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk.f8tobf16");
    Type retElemTy = i16_ty;
    int pk_vec = 2;
    Type dstInstrTy = vec_ty(retElemTy, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, i8_ty, retElemTy, i16_ty,
                               dstInstrTy, funcName);
  }

  static SmallVector<Value>
  Fp16_to_Fp8E5M2_RTNE_FAST(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk4.f16tobf8.cfg");
    Value rounding_mode = b.i32_val(0); // ``_MACA_FROUND_TO_NEAREST_INT``
    Value denormal_mode = b.i32_val(3); // ``_MACA_DENORMAL_FLUSH_NONE``
    Value src_modifiers = b.i32_val(0); // ``_MACA_SRC_NO_MODIFIER``
    Value clamp = b.i1_val(0);
    Value ovfl = b.i1_val(0);
    int pk_vec = 4;
    Type srcInstrTy = vec_ty(f16_ty, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, f16_ty, i8_ty, srcInstrTy,
                               i32_ty, funcName, true, pk_vec, rounding_mode,
                               denormal_mode, src_modifiers, clamp, ovfl);
  }

  static SmallVector<Value>
  Fp16_to_Fp8E4M3FN_RTNE_FAST(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &v,
                              triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk4.f16tof8.cfg");
    Value rounding_mode = b.i32_val(0); // ``_MACA_FROUND_TO_NEAREST_INT``
    Value denormal_mode = b.i32_val(3); // ``_MACA_DENORMAL_FLUSH_NONE``
    Value src_modifiers = b.i32_val(0); // ``_MACA_SRC_NO_MODIFIER``
    Value clamp = b.i1_val(0);
    Value ovfl = b.i1_val(0);
    int pk_vec = 4;
    Type srcInstrTy = vec_ty(f16_ty, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, f16_ty, i8_ty, srcInstrTy,
                               i32_ty, funcName, true, pk_vec, rounding_mode,
                               denormal_mode, src_modifiers, clamp, ovfl);
  }

  static SmallVector<Value>
  Fp16_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v, triton::FpToFpOp op) {
    SmallVector<Value> buf;
    for (Value val : v) {
      buf.push_back(convertFp16ToFp32(loc, rewriter, val));
    }
    return Fp32_to_Fp8E4M3FN_RTNE_SW(loc, rewriter, buf, op);
  }

  static SmallVector<Value>
  Bf16_to_Fp8E4M3FN_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v, triton::FpToFpOp op) {
    SmallVector<Value> buf;
    for (Value val : v) {
      buf.push_back(mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, val));
    }
    return Fp32_to_Fp8E4M3FN_RTNE_SW(loc, rewriter, buf, op);
  }

  static SmallVector<Value>
  Fp16_to_Fp8E5M2_RTNE_SW(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v, triton::FpToFpOp op) {
    SmallVector<Value> buf;
    for (Value val : v) {
      buf.push_back(convertFp16ToFp32(loc, rewriter, val));
    }
    return Fp32_to_Fp8E5M2_RTNE_SW(loc, rewriter, buf, op);
  }

  static SmallVector<Value>
  Fp8E4M3FN_to_Bf16_SW(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto buf = Fp8E4M3FN_to_Fp32(loc, rewriter, v, op);
    SmallVector<Value> ret;
    for (Value &val : buf) {
      ret.push_back(mlir::LLVM::METAX::convertFp32ToBf16(loc, op.getOperation(),
                                                         rewriter, val));
    }
    return ret;
  }

  static SmallVector<Value>
  Bf16_to_Fp8E4M3FN_FAST(Location loc, ConversionPatternRewriter &rewriter,
                         const SmallVector<Value> &v, triton::FpToFpOp op) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringRef funcName("llvm.mxc.cvt.pk.bf16tof8");
    Type srcElemTy = i16_ty;
    int pk_vec = 2;
    Type srcInstrTy = vec_ty(srcElemTy, pk_vec);
    return PK_Fp_to_Fp_Builtin(loc, rewriter, v, op, srcElemTy, i8_ty,
                               srcInstrTy, i16_ty, funcName);
  }

  static std::pair<ConverterT, size_t>
  Fp32_to_Fp8E5M2_RTNE(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F32_F8_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp32_to_Fp8E5M2_RTNE_HW, 4)
               : std::make_pair(Fp32_to_Fp8E5M2_RTNE_SW, 2);
  }

  static std::pair<ConverterT, size_t>
  Fp8E4M3FN_to_Fp16(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F8_F16_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp8E4M3FN_to_Fp16_HW, 2)
               : std::make_pair(Fp8E4M3FN_to_Fp16_SW, 2);
  }

  static std::pair<ConverterT, size_t> Fp8E5M2_to_Fp16(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F8_F16_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp8E5M2_to_Fp16_HW, 2)
               : std::make_pair(Fp8E5M2_to_Fp16_SW, 4);
  }

  static std::pair<ConverterT, size_t>
  Fp8E4M3FN_to_Bf16(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F8_F16_CVT") == nullptr;
    return (computeCapability >= 80 && enableFastCvt)
               ? std::make_pair(Fp8E4M3FN_to_Bf16_FAST, 2)
               : std::make_pair(Fp8E4M3FN_to_Bf16_SW, 2);
  }

  static std::pair<ConverterT, size_t>
  Fp16_to_Fp8E4M3FN(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F16_F8_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp16_to_Fp8E4M3FN_RTNE_FAST, 2)
               : std::make_pair(Fp16_to_Fp8E4M3FN_RTNE_SW, 2);
  }

  static std::pair<ConverterT, size_t> Fp16_to_Fp8E5M2(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F16_F8_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Fp16_to_Fp8E5M2_RTNE_FAST, 2)
               : std::make_pair(Fp16_to_Fp8E5M2_RTNE_SW, 2);
  }

  static std::pair<ConverterT, size_t>
  Bf16_to_Fp8E4M3FN(int computeCapability) {
    bool enableFastCvt =
        std::getenv("TRITON_DISABLE_FAST_F16_F8_CVT") == nullptr;
    return (computeCapability >= 89 && enableFastCvt)
               ? std::make_pair(Bf16_to_Fp8E4M3FN_FAST, 2)
               : std::make_pair(Bf16_to_Fp8E4M3FN_RTNE_SW, 2);
  }

  std::pair<ConverterT, size_t>
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

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>,
                    std::pair<ConverterT, size_t>>
        srcMap = {
            // F8 -> F16
            {{F8E4M3FNTyID, F16TyID, undefRounding},
             Fp8E4M3FN_to_Fp16(computeCapability)},
            {{F8E4M3FNUZTyID, F16TyID, undefRounding},
             std::make_pair(Fp8E4M3FNUZ_to_Fp16, 2)},
            {{F8E5M2TyID, F16TyID, undefRounding},
             Fp8E5M2_to_Fp16(computeCapability)},
            {{F8E5M2FNUZTyID, F16TyID, undefRounding},
             std::make_pair(Fp8E5M2FNUZ_to_Fp16, 2)},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding},
             std::make_pair(Fp8E5M2_to_Bf16, 4)},
            {{F8E4M3FNTyID, BF16TyID, undefRounding},
             Fp8E4M3FN_to_Bf16(computeCapability)},
            // F16 -> F8
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3FN(computeCapability)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2(computeCapability)},
            // BF16 -> F8
            {{BF16TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Bf16_to_Fp8E4M3FN(computeCapability)},
            // F32 <-> F8
            {{F32TyID, F8E4M3FNTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FN_RTNE(computeCapability)},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2_RTNE(computeCapability)},
            {{F32TyID, F8E4M3FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E4M3FNUZ_RTNE(computeCapability)},
            {{F32TyID, F8E5M2FNUZTyID, RoundingMode::RTNE},
             Fp32_to_Fp8E5M2FNUZ_RTNE(computeCapability)},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTZ},
             std::make_pair(Fp32ToFp8E5M2RTZ, 2)},
            {{F8E4M3FNUZTyID, F32TyID, undefRounding},
             std::make_pair(Fp8E4M3FNUZ_to_Fp32, 2)},
            {{F8E4M3FNTyID, F32TyID, undefRounding},
             std::make_pair(Fp8E4M3FN_to_Fp32, 2)},
            {{F8E5M2FNUZTyID, F32TyID, undefRounding},
             std::make_pair(Fp8E5M2FNUZ_to_Fp32, 2)},
            {{F8E5M2TyID, F32TyID, undefRounding},
             std::make_pair(Fp8E5M2_to_Fp32, 4)},
        };

    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    auto cvtPair = srcMap.lookup(key);
    return cvtPair;
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
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
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      outVals.reserve(operands[0].size());
      for (Value v : operands[0]) {
        outVals.push_back(mlir::LLVM::METAX::convertFp32ToBf16(
            loc, op.getOperation(), rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    auto enableFastF8Cvt =
        (computeCapability >= 89 && std::getenv("TRITON_ENABLE_FAST_FP8_CVT"));
    bool useFP16IntermediateSrc =
        srcElementType.isF32() && !(dstElementType.isFloat(8));
    bool useFP32IntermediateSrcF16 = srcElementType.isF16() &&
                                     (dstElementType.isFloat(8)) &&
                                     !(enableFastF8Cvt);
    bool useFP32IntermediateSrcBF16 = srcElementType.isBF16() &&
                                      (dstElementType.isFloat(8)) &&
                                      !(enableFastF8Cvt);
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    srcType = useFP32IntermediateSrcF16 ? f32_ty : srcType;
    srcType = useFP32IntermediateSrcBF16 ? f32_ty : srcType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, RoundingMode::RTZ);
    if (useFP32IntermediateSrcF16)
      for (Value &v : inVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    if (useFP32IntermediateSrcBF16)
      for (Value &v : inVals)
        v = mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, v);
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals, op);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                ValueRange operands) {
  auto v0 = mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0]);
  auto v1 = mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  return mlir::LLVM::METAX::convertFp32ToBf16(loc, result.getOperation(),
                                              rewriter, result);
}

template <typename OP>
Value EmitSingleFP16ElementwiseOp(Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  ValueRange operands) {
  auto v0 = FpToFpOpConversion::convertFp16ToFp32(loc, rewriter, operands[0]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0);
  return FpToFpOpConversion::convertFp32ToFp16(loc, rewriter, result);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    std::string instructionName;
    auto dtype = getElementType(op.getResult());
    bool isFp = dtype.isF32() || dtype.isF16() || dtype.isBF16();
    if (std::getenv("TRITON_ENABLE_RCP_DIVF") != nullptr) {
      instructionName = "llvm.mxc.rcp.f32";
      StringRef funcName(instructionName);
      if (dtype.isF32()) {
        ValueRange valueRange({operands[0][1]});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {
            rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0], ret)};
      } else if (dtype.isF16()) {
        auto v0 = FpToFpOpConversion::convertFp16ToFp32(loc, rewriter,
                                                        operands[0][0]);
        auto v1 = FpToFpOpConversion::convertFp16ToFp32(loc, rewriter,
                                                        operands[0][1]);
        ValueRange valueRange({v0, v1});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret_ = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                        .getResult();
        auto ret = FpToFpOpConversion::convertFp32ToFp16(loc, rewriter, ret_);
        return {
            rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0], ret)};
      } else {
        auto v0 =
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][0]);
        auto v1 =
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][1]);
        ValueRange valueRange({v0, v1});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret_ = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                        .getResult();
        auto ret = mlir::LLVM::METAX::convertFp32ToBf16(loc, op.getOperation(),
                                                        rewriter, ret_);
        return {
            rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0], ret)};
      }
    } else if (std::getenv("TRITON_ENABLE_FAST_DIVF") != nullptr) {
      instructionName = "llvm.mxc.fdiv.fast";
      StringRef funcName(instructionName);
      if (dtype.isF32()) {
        ValueRange valueRange({operands[0][0], operands[0][1]});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {ret};
      } else if (dtype.isF16()) {
        auto v0 = FpToFpOpConversion::convertFp16ToFp32(loc, rewriter,
                                                        operands[0][0]);
        auto v1 = FpToFpOpConversion::convertFp16ToFp32(loc, rewriter,
                                                        operands[0][1]);
        ValueRange valueRange({v0, v1});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {FpToFpOpConversion::convertFp32ToFp16(loc, rewriter, ret)};
      } else {
        auto v0 =
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][0]);
        auto v1 =
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][1]);
        ValueRange valueRange({v0, v1});
        Type funcType = mlir::triton::gpu::getFunctionType(dtype, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {mlir::LLVM::METAX::convertFp32ToBf16(loc, op.getOperation(),
                                                     rewriter, ret)};
      }
    } else {
      return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {
          EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands[0])};
    } else {
      if (std::getenv("TRITON_DISABLE_ELEMENTWISE_PK_FMA_OPT") == nullptr &&
          lhsElemTy.isF32() && rhsElemTy.isF32() && operands.size() >= 2) {
        auto vecType = vec_ty(f32_ty, 2);
        Value vec0 = b.undef(vecType);
        Value vec1 = b.undef(vecType);
        Value vec2 = b.undef(vecType);
        Value zero = b.f32_val(0.0);
        vec0 = b.insert_element(vec0, operands[0][0], b.i32_val(0));
        vec0 = b.insert_element(vec0, operands[1][0], b.i32_val(1));
        vec1 = b.insert_element(vec1, operands[0][1], b.i32_val(0));
        vec1 = b.insert_element(vec1, operands[1][1], b.i32_val(1));
        vec2 = b.insert_element(vec2, zero, b.i32_val(0));
        vec2 = b.insert_element(vec2, zero, b.i32_val(1));
        StringRef funcName("llvm.mxc.pk.fma.f32");
        ValueRange valueRange({vec0, vec1, vec2});
        Type funcType = mlir::triton::gpu::getFunctionType(vecType, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {b.extract_element(ret, b.i32_val(0)),
                b.extract_element(ret, b.i32_val(1))};
      }
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {
          EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands[0])};
    } else {
      if (std::getenv("TRITON_DISABLE_ELEMENTWISE_PK_FMA_OPT") == nullptr &&
          lhsElemTy.isF32() && rhsElemTy.isF32() && operands.size() >= 2) {
        auto vecType = vec_ty(f32_ty, 2);
        Value vec0 = b.undef(vecType);
        Value vec1 = b.undef(vecType);
        Value vec2 = b.undef(vecType);
        Value one = b.f32_val(1.0);
        vec0 = b.insert_element(vec0, operands[0][0], b.i32_val(0));
        vec0 = b.insert_element(vec0, operands[1][0], b.i32_val(1));
        vec1 = b.insert_element(vec1, one, b.i32_val(0));
        vec1 = b.insert_element(vec1, one, b.i32_val(1));
        vec2 = b.insert_element(vec2, operands[0][1], b.i32_val(0));
        vec2 = b.insert_element(vec2, operands[1][1], b.i32_val(1));
        StringRef funcName("llvm.mxc.pk.fma.f32");
        ValueRange valueRange({vec0, vec1, vec2});
        Type funcType = mlir::triton::gpu::getFunctionType(vecType, valueRange);
        LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
            rewriter, op, funcName, funcType);
        auto ret = LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange)
                       .getResult();
        return {b.extract_element(ret, b.i32_val(0)),
                b.extract_element(ret, b.i32_val(1))};
      }
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
      return {
          EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands[0])};
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<Value> createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      auto v0 =
          mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      auto v1 =
          mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][1]);
      auto result = rewriter.create<LLVM::FCmpOp>(
          loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()), v0, v1);
      return {result};
    } else {
      return {rewriter.create<LLVM::FCmpOp>(
          loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
          operands[0][0], operands[0][1])};
    }
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

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

    if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {mlir::LLVM::METAX::convertFp32ToBf16(loc, value.getOperation(),
                                                   rewriter, value)};
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
      auto value =
          mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][0]);
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
      return {
          mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, operands[0][0])};
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
      return {mlir::LLVM::METAX::convertFp32ToBf16(loc, op.getOperation(),
                                                   rewriter, operands[0][0])};
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
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    if (elemTy.getIntOrFloatBitWidth() == 32) {
      return {rewriter.create<LLVM::ExpOp>(loc, elemTy, operands[0])};
    } else if (elemTy.getIntOrFloatBitWidth() == 16) {
      return {
          EmitSingleFP16ElementwiseOp<LLVM::ExpOp>(loc, rewriter, operands[0])};
    } else {
      assert(false && "unsupported ExpOp operand type for MACA");
    }
  }
};

struct FRcpOpConversion
    : ElementwiseOpConversionBase<triton::RcpfOp, FRcpOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::RcpfOp, FRcpOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(triton::RcpfOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    std::string instructionName;
    auto dtype = getElementType(op.getSrc());
    if (dtype.isF16()) {
      instructionName = "llvm.mxc.rcp.f16";
    } else if (dtype.isF32()) {
      instructionName = "llvm.mxc.rcp.f32";
    } else {
      llvm_unreachable("rcpf dtype not supported.");
    }
    StringRef funcName(instructionName);
    ValueRange valueRange({op.getSrc()});
    auto ret = mlir::LLVM::createBuiltinFunc<triton::RcpfOp>(
        rewriter, loc, op, funcName, op.getResult().getType(), valueRange);
    return {ret};
  }
};

struct FmaOpConversion
    : ElementwiseOpConversionBase<mlir::math::FmaOp, FmaOpConversion> {
  using Base = ElementwiseOpConversionBase<mlir::math::FmaOp, FmaOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::FmaOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (std::getenv("TRITON_DISABLE_ELEMENTWISE_PK_FMA_OPT") == nullptr &&
        elemTy.isF32() && operands.size() >= 2) {
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      auto vecType = vec_ty(f32_ty, 2);
      Value vec0 = b.undef(vecType);
      Value vec1 = b.undef(vecType);
      Value vec2 = b.undef(vecType);
      vec0 = b.insert_element(vec0, operands[0][0], b.i32_val(0));
      vec0 = b.insert_element(vec0, operands[1][0], b.i32_val(1));
      vec1 = b.insert_element(vec1, operands[0][1], b.i32_val(0));
      vec1 = b.insert_element(vec1, operands[1][1], b.i32_val(1));
      vec2 = b.insert_element(vec2, operands[0][2], b.i32_val(0));
      vec2 = b.insert_element(vec2, operands[1][2], b.i32_val(1));
      StringRef funcName("llvm.mxc.pk.fma.f32");
      ValueRange valueRange({vec0, vec1, vec2});
      auto ret = mlir::LLVM::createBuiltinFunc<mlir::math::FmaOp>(
          rewriter, loc, op, funcName, vecType, valueRange);
      return {b.extract_element(ret, b.i32_val(0)),
              b.extract_element(ret, b.i32_val(1))};
    } else {
      return {};
    }
  }
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
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
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};

int getNumElementsPerThreads(Type type,
                             const LLVMTypeConverter *typeConverter) {
  int numElemsPerThread = 1;
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy)
    return numElemsPerThread;
  auto structType =
      dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
  assert(structType);
  numElemsPerThread = structType.getBody().size();
  return numElemsPerThread;
}

struct ElementwiseInlineIntrinsicOpConversion
    : public ConvertOpToLLVMPattern<ElementwiseInlineIntrinsicOp> {
  using Base = ConvertOpToLLVMPattern<ElementwiseInlineIntrinsicOp>;

  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ElementwiseInlineIntrinsicOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   const SmallVector<Value> &operands,
                                   Location loc) const {
    auto ctx = op->getContext();
    auto numResults = op.getResult().size();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Type> retTypes;
    for (auto result : op.getResult()) {
      auto ty = getTypeConverter()->convertType(getElementType(result));
      retTypes.push_back(ty);
    }
    if (!op.getNumResults())
      retTypes.push_back(void_ty(ctx));
    Type retTy = retTypes.size() > 1 ? struct_ty(retTypes) : retTypes[0];

    ValueRange operandsRange(operands);

    Value results =
        mlir::LLVM::createBuiltinFunc<triton::ElementwiseInlineIntrinsicOp>(
            rewriter, loc, op, op.getIntrinsicString(), retTy, operandsRange);

    SmallVector<Value> ret;
    for (int i = 0; i < op->getNumResults(); i++) {
      Value val;
      if (retTypes.size() > 1) {
        val = b.extract_val(results, i);
      } else {
        val = results;
      }
      ret.push_back(val);
    }
    return ret;
  }

  LogicalResult
  matchAndRewrite(ElementwiseInlineIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Layout is unpackedOperands[operand][elem].
    SmallVector<SmallVector<Value>> unpackedOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      unpackedOperands.push_back(subOperands);
    }

    int numElemsPerThread = getNumElementsPerThreads(op->getResult(0).getType(),
                                                     getTypeConverter());
    // Layout is results[result_idx][elem].
    //
    SmallVector<SmallVector<Value>> results(op->getNumResults());
    for (unsigned i = 0; i < numElemsPerThread; i++) {
      SmallVector<Value> block;
      for (auto &os : unpackedOperands) {
        block.push_back(os[i]);
      }
      auto cur = createDestOps(op, adaptor, rewriter, block, loc);
      assert(cur.size() == results.size());
      for (unsigned j = 0; j < cur.size(); j++) {
        results[j].push_back(cur[j]);
      }
    }

    // Reorder and pack the results.
    SmallVector<Value> outs;
    for (int i = 0; i < results.size(); i++) {
      outs.push_back(packLLElements(loc, getTypeConverter(), results[i],
                                    rewriter, op->getResult(i).getType()));
    }

    rewriter.replaceOp(op, outs);
    return success();
  }
};

} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::METAX::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "mc_math_func_sqrtf", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "mc_math_func_fdiv_rn", benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FRcpOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);

  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
#ifdef USE_MACA_DISTRIBUTED
  patterns.add<ElementwiseInlineIntrinsicOpConversion>(typeConverter, benefit);
#endif
#ifdef USE_MACA
  patterns.add<FmaOpConversion>(typeConverter, axisInfoAnalysis, benefit);
#endif
  bool hwNanPropagationSupported = false;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}
