// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Based on CANN 9.1 dav_c220/kernel_operator_vec_cmp_impl.h:
// CompareScalarCompute and VcmpvsIntrinsicsImpl. Keep the 252-repeat split
// so each packed-mask chunk starts on a 32-byte boundary for both dtypes.
template <typename T, typename U>
__aiv__ __attribute__((always_inline)) void
compare_impl(memref_t<__ubuf__ T, 1> *src, float scalar, int32_t comparison,
             memref_t<__ubuf__ U, 1> *dst) {
  auto input = src->aligned + src->offset;
  auto mask = reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned + dst->offset);
  T value = static_cast<T>(scalar);
  const uint32_t repeats = src->sizes[0] * sizeof(T) / 256;
  pipe_barrier(PIPE_V);
  for (uint32_t base = 0; base < repeats; base += 252) {
    const uint8_t count = repeats - base > 252 ? 252 : repeats - base;
    auto x = input + base * 256 / sizeof(T);
    auto y = mask + base * 256 / sizeof(T) / 8;
    if (comparison == 0)
      vcmpvs_eq(y, x, value, count, 1, 1, 8, 8);
    else if (comparison == 1)
      vcmpvs_gt(y, x, value, count, 1, 1, 8, 8);
    else
      vcmpvs_ge(y, x, value, count, 1, 1, 8, 8);
  }
  pipe_barrier(PIPE_V);
}

#define COMPARE_ENTRY(TYPE, SUFFIX, MASK)                                      \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_compare_scalar_##SUFFIX(                             \
          memref_t<__ubuf__ TYPE, 1> *src, float scalar,                       \
          memref_t<__ubuf__ MASK, 1> *dst) {                                   \
    compare_impl(src, scalar, 0, dst);                                         \
  }                                                                            \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_compare_scalar_##SUFFIX##_mode(                      \
          memref_t<__ubuf__ TYPE, 1> *src, float scalar, int32_t comparison,   \
          memref_t<__ubuf__ MASK, 1> *dst) {                                   \
    compare_impl(src, scalar, comparison, dst);                                \
  }
COMPARE_ENTRY(float, float, uint16_t)
COMPARE_ENTRY(half, half, uint16_t)
COMPARE_ENTRY(float, float_mask32, uint32_t)
