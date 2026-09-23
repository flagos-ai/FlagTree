// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

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
