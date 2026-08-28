//===------------------------- tx81.c--------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

bool is_dma_action_logging(uint32_t action) {
  return action & DMA_ACT_LOG;
}
bool is_dma_action_checking(uint32_t action) {
  return action & DMA_ACT_CHK;
}

uint64_t* get_header(const void *ddr_addr) {
  return (uint64_t*)((char*)ddr_addr - ClientPtrHeaderBytes);
}

uint32_t crc32(const uint8_t *data, int len)
{
  uint32_t crc = 0xFFFFFFFF;
  for (int i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++) {
      if (crc & 1)
        crc = (crc >> 1) ^ 0x04C11DB7;
      else
        crc >>= 1;
    }
  }
  return ~crc;
}

  

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

void debug_dump_f32_data(uint64_t *in, uint32_t elem_count, char *dump_flag) {
  RcsWaitfinish();
  volatile float* in_data = (volatile float*)get_spm_memory_mapping((uint64_t)in);
  __EP_LOG__(3, "------crt dump f32 data, addr: %p, dump_flag: %s, dump elem_count: %d\n",
             in_data, dump_flag, elem_count);
  for (int i = 0; i < elem_count; i++) {
    __EP_LOG__(3, "idx: %d, in: %.9f\n", i, in_data[i]);
  }
}

void debug_dump_i32_data(uint64_t *in, uint32_t elem_count, char *dump_flag) {
  RcsWaitfinish();
  volatile int32_t* in_data = (volatile int32_t*)get_spm_memory_mapping((uint64_t)in);
  __EP_LOG__(3, "------crt dump i32 data, addr: %p, dump_flag: %s, dump elem_count: %d\n",
             in_data, dump_flag, elem_count);
  for (int i = 0; i < elem_count; i++) {
    __EP_LOG__(3, "idx: %d, in: %d\n", i, in_data[i]);
  }
}

void debug_dump_i64_data(uint64_t *in, uint32_t elem_count, char *dump_flag) {
  RcsWaitfinish();
  volatile int64_t* in_data = (volatile int64_t*)get_spm_memory_mapping((uint64_t)in);
  __EP_LOG__(3, "------crt dump i32 data, addr: %p, dump_flag: %s, dump elem_count: %d\n",
             in_data, dump_flag, elem_count);
  for (int i = 0; i < elem_count; i++) {
    __EP_LOG__(3, "idx: %d, in: %lld\n", i, in_data[i]);
  }
}


float bf16_to_f32(uint16_t x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

void debug_dump_bf16_data(uint64_t *in, uint32_t elem_count, char *dump_flag) {
  RcsWaitfinish();
  volatile uint16_t* in_data = (volatile uint16_t*)get_spm_memory_mapping((uint64_t)in);
  __EP_LOG__(3, "------crt dump bf16 data, addr: %p, dump_flag: %s, dump elem_count: %d\n",
             in_data, dump_flag, elem_count);
  for (int i = 0; i < elem_count; i++) {
    float num = bf16_to_fp32(in_data[i]);
    __EP_LOG__(3, "idx: %d, in: %.9f\n", i, num);
  }
}

float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x03ff;

    uint32_t fbits;

    if (exp == 0) {
        if (frac == 0) {
            // zero
            fbits = sign;
        } else {
            // subnormal
            exp = 1;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03ff;

            uint32_t fexp = exp + (127 - 15);
            uint32_t ffrac = frac << 13;
            fbits = sign | (fexp << 23) | ffrac;
        }
    } else if (exp == 0x1f) {
        // inf / nan
        fbits = sign | 0x7f800000 | (frac << 13);
    } else {
        // normal
        uint32_t fexp = exp + (127 - 15);
        uint32_t ffrac = frac << 13;
        fbits = sign | (fexp << 23) | ffrac;
    }

    float f;
    memcpy(&f, &fbits, sizeof(f));
    return f;
}

void debug_dump_f16_data(uint64_t *in, uint32_t elem_count, char *dump_flag) {
  RcsWaitfinish();
  volatile uint16_t* in_data = (volatile uint16_t*)get_spm_memory_mapping((uint64_t)in);
  __EP_LOG__(3, "------crt dump f16 data, addr: %p, dump_flag: %s, dump elem_count: %d\n",
             in_data, dump_flag, elem_count);
  for (int i = 0; i < elem_count; i++) {
    float num = fp16_to_f32(in_data[i]);
    __EP_LOG__(3, "idx: %d, in: %.9f\n", i, num);
  }
}


static bool shape_has_zero_dim(const int *shape, int rank) {
  for (int i = 0; i < rank; ++i) {
    if (shape[i] == 0) {
      return true;
    }
  }
  return false;
}

static void offset_min_max_in_box(const int *shape, const int64_t *byte_stride,
                                  int rank, int64_t *min_off, int64_t *max_off) {
  int64_t lo = 0;
  int64_t hi = 0;
  for (int k = 0; k < rank; ++k) {
    assert(shape[k] > 0);
    int64_t last = shape[k] - 1;
    int64_t s = byte_stride[k];
    if (s >= 0) {
      hi += s * last;
    } else {
      lo += s * last;
    }
  }
  *min_off = lo;
  *max_off = hi;
}

static Tx81MemAddrRange mem_addr_range_per_element(uintptr_t base,
                                                 const int *shape,
                                                 const int64_t *byte_stride,
                                                 int rank,
                                                 uint32_t elem_bytes) {
  int64_t lo, hi;
  Tx81MemAddrRange range;
  offset_min_max_in_box(shape, byte_stride, rank, &lo, &hi);
  range.min_addr = (uintptr_t)((int64_t)base + lo);
  range.max_addr = (uintptr_t)((int64_t)base + hi + elem_bytes);
  return range;
}

static Tx81MemAddrRange mem_addr_range_inner_line(uintptr_t base,
                                                  const int *outer_shape,
                                                  const int64_t *byte_stride,
                                                  int rank,
                                                  int inner_dim_elems,
                                                  uint32_t elem_bytes) {
  Tx81MemAddrRange range;
  if (rank == 1) {
    range.min_addr = base;
    range.max_addr = base + (uint64_t)inner_dim_elems * elem_bytes;
    return range;
  }

  int64_t lo, hi;
  offset_min_max_in_box(outer_shape, byte_stride, rank - 1, &lo, &hi);
  range.min_addr = (uintptr_t)((int64_t)base + lo);
  range.max_addr = (uintptr_t)((int64_t)base + hi +
                               (int64_t)inner_dim_elems * elem_bytes);
  return range;
}

static Tx81MemAddrRange mem_addr_range_4d(uintptr_t base, const int *shape,
                                          const int64_t *byte_stride,
                                          uint32_t elem_bytes) {
  int64_t lo, hi;
  Tx81MemAddrRange range;
  offset_min_max_in_box(shape, byte_stride, 3, &lo, &hi);
  range.min_addr = (uintptr_t)((int64_t)base + lo);
  range.max_addr =
      (uintptr_t)((int64_t)base + hi + (int64_t)shape[3] * elem_bytes);
  return range;
}

Tx81MemAddrRange compute_rdma_src_addr_range(const void *src, int *src_shape,
                                             int *src_stride, int *dst_shape,
                                             int *dst_stride, int rank,
                                             uint32_t elem_bytes,
                                             uint32_t fmt) {
  uintptr_t base = (uintptr_t)src;
  Tx81MemAddrRange empty = {base, base};

  if (rank <= 0 || rank > TX81_MAX_TENSOR_RANK) {
    return empty;
  }
  if (shape_has_zero_dim(src_shape, rank)) {
    return empty;
  }

  int64_t byte_stride[TX81_MAX_TENSOR_RANK];
  for (int i = 0; i < rank; ++i) {
    byte_stride[i] = (int64_t)src_stride[i] * (int64_t)elem_bytes;
  }

  if (src_stride[rank - 1] != 1 || dst_stride[rank - 1] != 1) {
    return mem_addr_range_per_element(base, src_shape, byte_stride, rank,
                                    elem_bytes);
  }

  int shape_copy[TX81_MAX_TENSOR_RANK];
  int src_stride_copy[TX81_MAX_TENSOR_RANK];
  int dst_shape_copy[TX81_MAX_TENSOR_RANK];
  int dst_stride_copy[TX81_MAX_TENSOR_RANK];
  memcpy(shape_copy, src_shape, sizeof(int) * rank);
  memcpy(src_stride_copy, src_stride, sizeof(int) * rank);
  memcpy(dst_shape_copy, dst_shape, sizeof(int) * rank);
  memcpy(dst_stride_copy, dst_stride, sizeof(int) * rank);

  uint32_t elem_bytes_eff = elem_bytes;
  uint32_t fmt_eff = fmt;
  legalizeMemoryOpAttribute(shape_copy, src_stride_copy, dst_shape_copy,
                            dst_stride_copy, rank, &elem_bytes_eff, &fmt_eff);
  (void)fmt_eff;

  for (int i = 0; i < rank; ++i) {
    byte_stride[i] = (int64_t)src_stride_copy[i] * (int64_t)elem_bytes_eff;
  }

  if (rank == 4 && no_reverse_memory_access(src_stride_copy, rank) &&
      is_contiguous(dst_shape_copy, dst_stride_copy, rank)) {
    return mem_addr_range_4d(base, shape_copy, byte_stride, elem_bytes_eff);
  }

  return mem_addr_range_inner_line(base, shape_copy, byte_stride, rank,
                                   shape_copy[rank - 1], elem_bytes_eff);
}

Tx81MemAddrRange compute_wdma_dst_addr_range(const void *dst, int *src_shape,
                                             int *src_stride, int *dst_shape,
                                             int *dst_stride, int rank,
                                             uint32_t elem_bytes,
                                             uint32_t fmt) {
  uintptr_t base = (uintptr_t)dst;
  Tx81MemAddrRange empty = {base, base};

  if (rank <= 0 || rank > TX81_MAX_TENSOR_RANK) {
    return empty;
  }
  if (shape_has_zero_dim(src_shape, rank)) {
    return empty;
  }

  int64_t byte_stride[TX81_MAX_TENSOR_RANK];
  for (int i = 0; i < rank; ++i) {
    byte_stride[i] = (int64_t)dst_stride[i] * (int64_t)elem_bytes;
  }

  if (src_stride[rank - 1] != 1 || dst_stride[rank - 1] != 1) {
    return mem_addr_range_per_element(base, src_shape, byte_stride, rank,
                                      elem_bytes);
  }

  int src_shape_copy[TX81_MAX_TENSOR_RANK];
  int src_stride_copy[TX81_MAX_TENSOR_RANK];
  int dst_shape_copy[TX81_MAX_TENSOR_RANK];
  int dst_stride_copy[TX81_MAX_TENSOR_RANK];
  memcpy(src_shape_copy, src_shape, sizeof(int) * rank);
  memcpy(src_stride_copy, src_stride, sizeof(int) * rank);
  memcpy(dst_shape_copy, dst_shape, sizeof(int) * rank);
  memcpy(dst_stride_copy, dst_stride, sizeof(int) * rank);

  uint32_t elem_bytes_eff = elem_bytes;
  uint32_t fmt_eff = fmt;
  legalizeMemoryOpAttribute(src_shape_copy, src_stride_copy, dst_shape_copy,
                            dst_stride_copy, rank, &elem_bytes_eff, &fmt_eff);
  (void)fmt_eff;

  for (int i = 0; i < rank; ++i) {
    byte_stride[i] = (int64_t)dst_stride_copy[i] * (int64_t)elem_bytes_eff;
  }

  if (rank == 4 && no_reverse_memory_access(dst_stride_copy, rank) &&
      is_contiguous(src_shape_copy, src_stride_copy, rank)) {
    return mem_addr_range_4d(base, dst_shape_copy, byte_stride, elem_bytes_eff);
  }

  return mem_addr_range_inner_line(base, src_shape_copy, byte_stride, rank,
                                   src_shape_copy[rank - 1], elem_bytes_eff);
}

// --- DMA bounds check counters ---

uint32_t dma_oob_count = 0;
uint32_t dma_bad_magic_count = 0;

static bool dma_check_abort_enabled = false;

bool get_dma_check_abort(void) {
  return dma_check_abort_enabled;
}

__attribute__((constructor))
static void dma_check_init(void) {
  const char *val = getenv("TRITON_DMA_CHECK_ABORT");
  if (val && val[0] == '1') {
    dma_check_abort_enabled = true;
  }
}

uint32_t get_dma_oob_count(void) {
  return dma_oob_count;
}

uint32_t get_dma_bad_magic_count(void) {
  return dma_bad_magic_count;
}

void reset_dma_oob_count(void) {
  dma_oob_count = 0;
}

void reset_dma_bad_magic_count(void) {
  dma_bad_magic_count = 0;
}

#ifdef __cplusplus
}
#endif
