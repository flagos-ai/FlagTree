//===------------------------- tx81.c--------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

bool is_contiguous(int *shape, int *strides, int rank) {
  int expected_stride = 1;
  for (int i = rank - 1; i >= 0; i--) {
    if (shape[i] != 1 && strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= shape[i];
  }
  return true;
}
uint64_t next_power_of_two_64(uint64_t x) {
  if (x == 0) {
    return 1;
  }
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

uint32_t get_dtype_size_new(Data_Format fmt) {
  switch (fmt) {
  case Fmt_INT8:
    return sizeof(int8_t);
  case Fmt_INT16:
  case Fmt_FP16:
  case Fmt_BF16:
    return sizeof(int16_t);
  case Fmt_INT32:
  case Fmt_FP32:
  case Fmt_TF32:
    return sizeof(int32_t);
  case Fmt_INT64:
    return sizeof(int64_t);
  default:
    assert(false && "Unsupported format\n");
    return 0;
  }
}

uint32_t get_cx_align_base_new(uint32_t c, Data_Format fmt) {
  switch (fmt) {
  case Fmt_INT8:
    return c < 128 ? (c < 4 ? 4 : next_power_of_two_64(c)) : 128;
  case Fmt_INT16:
  case Fmt_FP16:
  case Fmt_BF16:
  case Fmt_INT32:
  case Fmt_FP32:
  case Fmt_TF32:
    return c < 64 ? (c < 4 ? 4 : next_power_of_two_64(c)) : 64;
  default:
    assert(false && "Unsupported format\n");
    return 0;
  }
}

bool no_reverse_memory_access(int *stride, int rank) {
  for (int i = 1; i < rank; i++) {
    if (stride[i] < 0) {
      return false;
    }
  }
  return true;
}

void tx81_memcpy(char *srcPtr, char *dstPtr, int *src_shape, int *src_stride,
                 int *dst_shape, int *dst_stride, int rank,
                 uint32_t elem_bytes) {
  int64_t readIndex = 0;
  int64_t writeIndex = 0;
  int64_t indices[rank], srcStrides[rank], dstStrides[rank];

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = (int64_t)src_stride[rankp] * (int64_t)elem_bytes;
    dstStrides[rankp] = (int64_t)dst_stride[rankp] * (int64_t)elem_bytes;
  }

  for (;;) {
    // Copy over the element, byte by byte.
    for (int i = 0; i < elem_bytes; i++)
      dstPtr[writeIndex + i] = srcPtr[readIndex + i];

    // Advance index and read position.
    // Loop from innermost dimension
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      int64_t newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src_shape[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= newIndex * srcStrides[axis];
      writeIndex -= newIndex * dstStrides[axis];
    }
  }
}

void legalizeMemoryOpAttribute(int *src_shape, int *src_stride, int *dst_shape,
                               int *dst_stride, int rank, uint32_t *elem_bytes,
                               uint32_t *fmt) {
  switch (*fmt) {
  case Fmt_INT8: {
    break;
  }
  case Fmt_INT16:
  case Fmt_FP16:
  case Fmt_BF16: {
    *fmt = Fmt_FP16;
    break;
  }
  case Fmt_INT32:
  case Fmt_FP32:
  case Fmt_TF32: {
    *fmt = Fmt_FP32;
    break;
  }
  case Fmt_INT64: {
    *fmt = Fmt_FP32;
    src_shape[rank - 1] *= sizeof(int64_t) / sizeof(int32_t);
    dst_shape[rank - 1] *= sizeof(int64_t) / sizeof(int32_t);
    *elem_bytes = sizeof(int32_t);
    // Last stride is always 1
    for (int i = 0; i < rank - 1; i++) {
      src_stride[i] *= 2;
      dst_stride[i] *= 2;
    }
    break;
  }
  default: {
    // Other formats are not supported.
    assert(false && "Unsupported format\n");
    break;
  }
  }
}

// Used for kcore load/store data from/to spm
const int64_t spmMappingOffset = 0x30400000;

int8_t *get_spm_memory_mapping_wrapper(uint64_t ptr) {
#ifdef USE_SIM_MODE
  return get_spm_memory_mapping(ptr);
#else
  return (int8_t *)(ptr + spmMappingOffset);
#endif
}

#ifdef __cplusplus
}
#endif
