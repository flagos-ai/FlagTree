#include "Utils.h"

__aiv__ __attribute__((always_inline)) void
check_inputs_of_gather_mask(const uint8_t src1_pattern) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(((src1_pattern >= 1) && (src1_pattern <= 7)) &&
         "gather_mask: src1_pattern must be in range [1, 7]");
#endif
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_mask_impl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
                 const uint8_t pattern_mode, const bool reduce_mode,
                 const uint32_t mask, const uint8_t src0_block_stride,
                 const uint16_t repeat_times, const uint16_t src0_repeat_stride,
                 const uint8_t src1_repeat_stride, __ubuf__ int64_t *rsvd_cnt) {
  if (reduce_mode) {
    set_mask_count();
  } else {
    set_mask_norm();
  }
  set_vector_mask(0, mask);
  vreducev2(dst, src0, src1, repeat_times, src0_block_stride, pattern_mode,
            src0_repeat_stride, src1_repeat_stride);
  *rsvd_cnt = get_rsvd_cnt();
  set_mask_norm();
}

#define REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(TYPE_NAME, T)              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_gather_mask_builtin_pattern_##TYPE_NAME(             \
          memref_t<__ubuf__ T, 1> *src0, const uint8_t src1_pattern,           \
          const bool reduce_mode, const uint32_t mask,                         \
          const uint8_t src0_block_stride, const uint16_t repeat_times,        \
          const uint16_t src0_repeat_stride, const uint8_t src1_repeat_stride, \
          memref_t<__ubuf__ T, 1> *dst,                                        \
          memref_t<__ubuf__ int64_t, 1> *rsvd_cnt) {                           \
    const int32_t ONE_REPEAT_BYTE_SIZE = 256;                                  \
    auto src0_ptr = src0->aligned + src0->offset;                              \
    auto null_src1_ptr = ONE_REPEAT_BYTE_SIZE * sizeof(T) + src0_ptr;          \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    auto rsvd_cnt_ptr = rsvd_cnt->aligned + rsvd_cnt->offset;                  \
    check_inputs_of_gather_mask(src1_pattern);                                 \
    if (sizeof(T) == sizeof(uint16_t)) {                                       \
      gather_mask_impl(reinterpret_cast<__ubuf__ uint16_t *>(dst_ptr),         \
                       reinterpret_cast<__ubuf__ uint16_t *>(src0_ptr),        \
                       reinterpret_cast<__ubuf__ uint16_t *>(null_src1_ptr),   \
                       src1_pattern, reduce_mode, mask, src0_block_stride,     \
                       repeat_times, src0_repeat_stride, src1_repeat_stride,   \
                       rsvd_cnt_ptr);                                          \
    } else {                                                                   \
      gather_mask_impl(reinterpret_cast<__ubuf__ uint32_t *>(dst_ptr),         \
                       reinterpret_cast<__ubuf__ uint32_t *>(src0_ptr),        \
                       reinterpret_cast<__ubuf__ uint32_t *>(null_src1_ptr),   \
                       src1_pattern, reduce_mode, mask, src0_block_stride,     \
                       repeat_times, src0_repeat_stride, src1_repeat_stride,   \
                       rsvd_cnt_ptr);                                          \
    }                                                                          \
  }

#define REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(TYPE_NAME, T, U)            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_gather_mask_custom_pattern_##TYPE_NAME(              \
          memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ U, 1> *src1,        \
          const bool reduce_mode, const uint32_t mask,                         \
          const uint8_t src0_block_stride, const uint16_t repeat_times,        \
          const uint16_t src0_repeat_stride, const uint8_t src1_repeat_stride, \
          memref_t<__ubuf__ T, 1> *dst,                                        \
          memref_t<__ubuf__ int64_t, 1> *rsvd_cnt) {                           \
    auto src0_ptr = src0->aligned + src0->offset;                              \
    auto src1_ptr = src1->aligned + src1->offset;                              \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    auto rsvd_cnt_ptr = rsvd_cnt->aligned + rsvd_cnt->offset;                  \
    gather_mask_impl(reinterpret_cast<__ubuf__ U *>(dst_ptr),                  \
                     reinterpret_cast<__ubuf__ U *>(src0_ptr), src1_ptr, 0,    \
                     reduce_mode, mask, src0_block_stride, repeat_times,       \
                     src0_repeat_stride, src1_repeat_stride, rsvd_cnt_ptr);    \
  }

extern "C" {
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(half, half)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(bf16, bfloat16_t)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(ushort, uint16_t)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(short, int16_t)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(float, float)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(uint, uint32_t)
REGISTER_CIFACE_GATHER_MASK_BUILTIN_PATTERN(int, int32_t)

REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(half, half, uint16_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(bf16, bfloat16_t, uint16_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(ushort, uint16_t, uint16_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(short, int16_t, uint16_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(float, float, uint32_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(uint, uint32_t, uint32_t)
REGISTER_CIFACE_GATHER_MASK_CUSTOM_PATTERN(int, int32_t, uint32_t)
}
