#include "Utils.h"

__aiv__ static inline __attribute__((always_inline)) void
set_mask(int32_t len, int32_t dtype_size) {
  const int32_t DEFAULT_BLOCK_SIZE = 256;
  const uint64_t FULL_MASK = 0xffffffffffffffff;
  int32_t type_len = DEFAULT_BLOCK_SIZE / dtype_size;
  const int32_t half_type_len = 64; // 1 register -> 64 bits -> 64 elements
  const int32_t len_coeff = 2;      // 2 registers for masks
  if (len == half_type_len) {
    set_vector_mask(0, FULL_MASK);
    return;
  } else if (len == type_len || len >= half_type_len * len_coeff) {
    // len = max ele per repeat / len >= 128
    set_vector_mask(FULL_MASK, FULL_MASK);
    return;
  }
  set_vector_mask(static_cast<uint64_t>(
                      (len > half_type_len)
                          ? (((static_cast<uint64_t>(1))
                              << static_cast<uint32_t>(len - half_type_len)) -
                             1)
                          : 0),
                  static_cast<uint64_t>((len > half_type_len)
                                            ? FULL_MASK
                                            : (((static_cast<uint64_t>(1))
                                                << static_cast<uint32_t>(len)) -
                                               1)));
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
pair_reduce_sum_impl(__ubuf__ T *dst_local, __ubuf__ T *src_local,
                     const int32_t repeat_times, const int32_t mask,
                     const int32_t dst_rep_stride, const int32_t src_blk_stride,
                     const int32_t src_rep_stride) {
  set_mask(mask, sizeof(T));
  vcpadd(dst_local, src_local, repeat_times, dst_rep_stride, src_blk_stride,
         src_rep_stride);
}

#define REGISTER_CIFACE_PAIR_REDUCE_SUM_CONTINUOUS_MASK(TYPE_NAME, T)          \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_pair_reduce_sum_continuous_mask_##TYPE_NAME(         \
          memref_t<__ubuf__ T, 1> *src, const int32_t repeat_times,            \
          const int32_t mask, const int32_t dst_rep_stride,                    \
          const int32_t src_blk_stride, const int32_t src_rep_stride,          \
          memref_t<__ubuf__ T, 1> *dst) {                                      \
    auto src_ptr = src->aligned + src->offset;                                 \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    pair_reduce_sum_impl(dst_ptr, src_ptr, repeat_times, mask, dst_rep_stride, \
                         src_blk_stride, src_rep_stride);                      \
  }

extern "C" {
REGISTER_CIFACE_PAIR_REDUCE_SUM_CONTINUOUS_MASK(half, half)
REGISTER_CIFACE_PAIR_REDUCE_SUM_CONTINUOUS_MASK(float, float)
}
