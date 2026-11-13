#include "Utils.h"

#if defined(__DAV_C310__)
// A5 (c310) is reg-based; the legacy mem-based vector_dup/set_vector_mask
// intrinsics are not available. Use the arch-portable AscendC Level-0
// duplicate entry, which selects the RegBase implementation (dav_3510) at
// compile time.
#include "kernel_operator_vec_duplicate_intf.h"
#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void duplicate_bitwise_mask_impl(
    __ubuf__ T *dst, T scalar, __ubuf__ uint64_t *mask, const uint8_t repeat_times,
    const uint16_t dst_block_stride, const uint8_t dst_repeat_stride) {
#if defined(__DAV_C310__)
  // Bit-mask mode: the A5 entry copies the 128-bit mask into a Reg::MaskReg.
  uint64_t mask_array[4] = {mask[0], mask[1], 0, 0};
  AscendC::DuplicateImpl<T>(dst, scalar, mask_array, repeat_times,
                            dst_block_stride, dst_repeat_stride);
#else
  set_vector_mask(static_cast<uint64_t>(mask[1]),
                  static_cast<uint64_t>(mask[0]));
  vector_dup(dst, scalar, repeat_times, dst_block_stride, 1, dst_repeat_stride,
             0);
#endif
}

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
    duplicate_bitwise_mask_impl<T>(dst_ptr, static_cast<T>(scalar_ptr[0]),     \
                                   mask_ptr, repeat_times, dst_block_stride,   \
                                   dst_repeat_stride);                         \
  }

extern "C" {
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(half, half)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(bf16, bfloat16_t)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(float, float)
}
