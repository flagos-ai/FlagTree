//===------------------------- tx81.c--------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

bool is_contiguous(int *shape, int *strides, int elem_bytes) {
  int expected_stride = elem_bytes;
  for (int i = 0; i < 4; i++) {
    if (strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= shape[i];
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
