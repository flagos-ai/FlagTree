// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

#if defined(__DAV_C310__)
// A5 compare is reg-based; the mem-based vcmpvs_* intrinsics are unavailable.
#include "kernel_operator_vec_cmpsel_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"

// A5 comparison modes treat NaN differently from the portable op (LE/GE and
// possibly others are unordered, i.e. NaN compares true). Normalize every mode
// to ordered semantics by AND-ing the comparison with (x == x): EQ with itself
// is false for NaN and true otherwise.
template <typename T, AscendC::CMPMODE cmpMode>
__simd_vf__ inline void compare_scalar_ordered(__ubuf__ uint8_t *dst,
                                               __ubuf__ T *src, T scalar,
                                               uint32_t count) {
  constexpr uint32_t elems = AscendC::GetVecLen() / sizeof(T);
  uint32_t sreg = count;
  AscendC::Reg::RegTensor<T> x;
  AscendC::Reg::UnalignReg ureg;
  for (int64_t i = 0; i < count; i += elems) {
    AscendC::Reg::MaskReg active = AscendC::Reg::UpdateMask<T>(sreg);
    AscendC::Reg::LoadAlign(x, src + i);
    AscendC::Reg::MaskReg cmp, not_nan, result;
    AscendC::Reg::CompareScalar<T, cmpMode>(cmp, x, scalar, active);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::EQ>(not_nan, x, x, active);
    AscendC::Reg::And(result, cmp, not_nan, active);
    // The mask store steps the destination pointer in T-sized elements (the
    // AscendC Level-2 compare does the same); using uint32_t here corrupts the
    // packing for 16-bit dtypes.
    AscendC::Reg::StoreUnAlign((__ubuf__ T *&)dst, result, ureg);
  }
  AscendC::Reg::StoreUnAlignPost<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_NORMAL>(
      dst, ureg, 0);
}

// Every IEEE comparison against a NaN scalar is false. A5 CompareScalar is
// unordered for LT/GT/LE/GE/NE, but EQ is ordered, so comparing against the
// NaN scalar yields the required all-zero mask.
template <typename T>
__simd_vf__ inline void compare_scalar_nan(__ubuf__ uint8_t *dst,
                                           __ubuf__ T *src, T scalar,
                                           uint32_t count) {
  constexpr uint32_t elems = AscendC::GetVecLen() / sizeof(T);
  uint32_t sreg = count;
  AscendC::Reg::RegTensor<T> x;
  AscendC::Reg::UnalignReg ureg;
  for (int64_t i = 0; i < count; i += elems) {
    AscendC::Reg::MaskReg active = AscendC::Reg::UpdateMask<T>(sreg);
    AscendC::Reg::LoadAlign(x, src + i);
    AscendC::Reg::MaskReg result;
    AscendC::Reg::CompareScalar<T, AscendC::CMPMODE::EQ>(result, x, scalar,
                                                         active);
    AscendC::Reg::StoreUnAlign((__ubuf__ T *&)dst, result, ureg);
  }
  AscendC::Reg::StoreUnAlignPost<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_NORMAL>(
      dst, ureg, 0);
}
#endif

__aiv__ __attribute__((always_inline)) void
check_mode_of_compare_scalar(const int32_t cmp_mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(((cmp_mode >= 0) && (cmp_mode <= 5)) &&
         "compare_scalar: cmp_mode must be a CANN CMPMODE value in [0, 5]");
#endif
}

// Based on CANN 9.1 dav_c220/kernel_operator_vec_cmp_impl.h:
// VcmpvsIntrinsicsImpl with the default unary repeat params
// {dstBlkStride=1, srcBlkStride=1, dstRepStride=8, srcRepStride=8}.
// cmp_mode is the CANN CMPMODE value from
// impl/basic_api/utils/kernel_utils_mode.h: 0=LT, 1=GT, 2=EQ, 3=LE, 4=GE,
// 5=NE.
template <typename T>
__aiv__ __attribute__((always_inline)) void
compare_scalar_vcmpvs(__ubuf__ uint8_t *dst, __ubuf__ T *src0, T src1,
                      const int32_t cmp_mode, const uint8_t repeat_time) {
  switch (cmp_mode) {
  case 0:
    vcmpvs_lt(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  case 1:
    vcmpvs_gt(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  case 2:
    vcmpvs_eq(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  case 3:
    vcmpvs_le(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  case 4:
    vcmpvs_ge(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  case 5:
    vcmpvs_ne(dst, src0, src1, repeat_time, 1, 1, 8, 8);
    break;
  default:
    break;
  }
}

// Based on CANN 9.1 dav_c220/kernel_operator_vec_cmp_impl.h:
// CompareScalarCompute (CompareScalar Level 2). Keep the 252-repeat split so
// every packed-mask chunk starts on a 32-byte boundary for both dtypes.
// Pipeline barriers are the caller's responsibility; this op adds none.
template <typename T, typename U>
__aiv__ __attribute__((always_inline)) void
compare_scalar_impl(memref_t<__ubuf__ T, 1> *src, float scalar,
                    int32_t cmp_mode, uint32_t count,
                    memref_t<__ubuf__ U, 1> *dst) {
#if defined(__DAV_C310__)
  // Count-based ordered compare; the A5 implementation packs the mask with
  // Reg::CompareScalar/StoreUnAlign, so no manual repeat splitting is needed.
  auto dst_byte =
      reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned + dst->offset);
  auto src_ptr = src->aligned + src->offset;
  const T scalar_value = static_cast<T>(scalar);
  // Data-dependent control flow inside a __simd_vf__ function crashes the
  // ccec VF compiler, so route the NaN scalar case here instead.
  if (scalar != scalar) {
    compare_scalar_nan<T>(dst_byte, src_ptr, scalar_value, count);
    return;
  }
  switch (cmp_mode) {
  case 0:
    compare_scalar_ordered<T, AscendC::CMPMODE::LT>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  case 1:
    compare_scalar_ordered<T, AscendC::CMPMODE::GT>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  case 2:
    compare_scalar_ordered<T, AscendC::CMPMODE::EQ>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  case 3:
    compare_scalar_ordered<T, AscendC::CMPMODE::LE>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  case 4:
    compare_scalar_ordered<T, AscendC::CMPMODE::GE>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  case 5:
    compare_scalar_ordered<T, AscendC::CMPMODE::NE>(dst_byte, src_ptr,
                                                    scalar_value, count);
    break;
  default:
    break;
  }
#else
  auto src0 = src->aligned + src->offset;
  auto dst_ptr =
      reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned + dst->offset);
  const T src1 = static_cast<T>(scalar);
  const uint32_t sum_repeat = count * sizeof(T) / 256;
  const uint32_t repeat_round = sum_repeat / 252;
  const uint32_t repeat_tail = sum_repeat % 252;
  const uint32_t src_offset = 252 * 256 / sizeof(T);
  const uint32_t dst_offset = src_offset / 8;
  for (uint32_t i = 0; i < repeat_round; ++i) {
    compare_scalar_vcmpvs(dst_ptr + i * dst_offset, src0 + i * src_offset, src1,
                          cmp_mode, 252);
  }
  compare_scalar_vcmpvs(dst_ptr + repeat_round * dst_offset,
                        src0 + repeat_round * src_offset, src1, cmp_mode,
                        static_cast<uint8_t>(repeat_tail));
#endif
}

#define REGISTER_CIFACE_COMPARE_SCALAR(TYPE_NAME, T, U)                        \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_compare_scalar_##TYPE_NAME(                          \
          memref_t<__ubuf__ T, 1> *src0, float src1, int32_t cmp_mode,         \
          uint32_t count, memref_t<__ubuf__ U, 1> *dst) {                      \
    check_mode_of_compare_scalar(cmp_mode);                                    \
    compare_scalar_impl(src0, src1, cmp_mode, count, dst);                     \
  }

REGISTER_CIFACE_COMPARE_SCALAR(half, half, uint16_t)
REGISTER_CIFACE_COMPARE_SCALAR(float, float, uint16_t)
REGISTER_CIFACE_COMPARE_SCALAR(float_mask32, float, uint32_t)
