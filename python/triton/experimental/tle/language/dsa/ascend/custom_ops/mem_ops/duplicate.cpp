#include "Utils.h"

#define REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(TYPE_NAME, T)                   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_duplicate_bitwise_mask_##TYPE_NAME(                  \
          memref_t<__ubuf__ T, 1> *scalar_value,                               \
          memref_t<__ubuf__ uint64_t, 1> *mask, const uint8_t repeat_times,    \
          const uint16_t dst_block_stride, const uint8_t dst_repeat_stride,    \
          memref_t<__ubuf__ T, 1> *dst) {                                      \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    auto scalar_ptr = scalar_value->aligned + scalar_value->offset;            \
    auto mask_ptr = mask->aligned + mask->offset;                              \
    set_vector_mask(static_cast<uint64_t>(mask_ptr[1]),                        \
                    static_cast<uint64_t>(mask_ptr[0]));                       \
    vector_dup(dst_ptr, static_cast<T>(scalar_ptr[0]), repeat_times,           \
               dst_block_stride, 1, dst_repeat_stride, 0);                     \
  }

extern "C" {
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(half, half)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(bf16, bfloat16_t)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(float, float)
}
