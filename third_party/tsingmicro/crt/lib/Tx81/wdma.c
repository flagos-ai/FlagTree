//===------------------------ wdma.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Wdma, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"
#include <stdio.h>
#include <stdlib.h>

void __Wdma4d(void *restrict dest, const void *restrict src,
              uint32_t elem_count, uint32_t stride0, uint32_t iteration0,
              uint32_t stride1, uint32_t iteration1, uint32_t stride2,
              uint32_t iteration2, uint32_t fmt) {
  INTRNISIC_RUN_SWITCH;
  RcsWdma *wdma = g_intrinsic()->wdma_pointer;
  RcsWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  wdma->AddSrcDst(&inst, (uint64_t)src, (uint64_t)dest, (Data_Format)fmt);
  wdma->ConfigStrideIteration(&inst, elem_count, stride0, iteration0, stride1,
                              iteration1, stride2, iteration2);
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
}

void __Wdma1d(void *restrict dest, const void *restrict src,
              uint32_t elem_count, uint32_t fmt, uint32_t action,
              const void *base_dst, const char *kernel_name) {
  INTRNISIC_RUN_SWITCH;

  if (is_dma_action_logging(action)) {
    __EP_LOG__(
        3, "[func @%s] wdma1d -- src: %p, dst: %p, elem_count: %d, fmt: %d\n",
        kernel_name, src, dest, elem_count, fmt);
  }

  if (is_dma_action_checking(action)) {
    uint64_t *header = get_header(base_dst);
    uint64_t CRC =
        crc32((uint8_t *)header, ClientPtrHeaderBytes - sizeof(uint64_t));
    if (CRC != header[2]) {
      __EP_LOG__(3,
                 "\n\nerror: [func @%s] wdma total_size %llu, magic %x, bad "
                 "crc %x != %x(header[2]), "
                 "base_dst (%p) data %x, src (%p)\n\n",
                 kernel_name, header[0], header[1], CRC, header[2], base_dst,
                 header[2], src);
    } else if (header[1] != ClientPtrMagic) {
      dma_bad_magic_count++;
      __EP_LOG__(3,
                 "\n\nerror: [func @%s] total_size %llu, bad magic %x, crc %x, "
                 "base_dst (%p) data %x, dst (%p)\n\n",
                 kernel_name, header[0], header[1], header[2], base_dst,
                 header[2], dest);
    } else {
      uint64_t total_size = header[0];
      uint32_t elem_bytes = get_dtype_size_new((Data_Format)fmt);
      uintptr_t min_addr = (uintptr_t)dest;
      uintptr_t max_addr = (uintptr_t)dest + (uintptr_t)elem_count * elem_bytes;

      __EP_LOG__(
          3,
          "[func @%s] wdma1d base_dst (%p), total_size %llu, magic %x, crc %x, "
          "dst (%p), range_size %d\n",
          kernel_name, base_dst, total_size, header[1], header[2], dest,
          (unsigned int)(max_addr - min_addr));

      if (min_addr < (uintptr_t)base_dst ||
          max_addr > (uintptr_t)base_dst + total_size) {
        dma_oob_count++;
        __EP_LOG__(3,
                   "[func @%s] fatal error: ddr memory OOB, "
                   "wdma1d base_dst (%p), total_size %llu, dst (%p), "
                   "dma_oob_count (%d)\n",
                   kernel_name, base_dst, total_size, dest, dma_oob_count);
        if (get_dma_check_abort()) {
          abort();
        }
      }
    }
  }

  RcsWdma *wdma = g_intrinsic()->wdma_pointer;
  RcsWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};
  wdma->Wdma1d(&inst, (uint64_t)src, (uint64_t)dest, elem_count,
               (Data_Format)fmt);
  RcsExecute(&inst);
  // RcsWaitfinish();
  SYNCHRONOUS_INTRINSIC_SWITCH;
}

// Wdma line by line.
void __WdmaVectorize(char *srcPtr, char *dstPtr, int *src_shape,
                     int *src_stride, int *dst_shape, int *dst_stride, int rank,
                     uint32_t elem_bytes, uint32_t fmt, int innermost_rank,
                     int inner_elem_count) {
  INTRNISIC_RUN_SWITCH;
  RcsWdma *wdma = g_intrinsic()->wdma_pointer;
  RcsWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

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
    // Copy inner dim, line by line.
    wdma->Wdma1d(&inst, (uint64_t)(srcPtr + readIndex),
                 (uint64_t)(dstPtr + writeIndex), inner_elem_count,
                 (Data_Format)fmt);
    RcsExecute(&inst);
    // RcsWaitfinish();
    SYNCHRONOUS_INTRINSIC_SWITCH;

    // Advance index and read position.
    // Start from the second-to-last dimension, copy one line at a time
    for (int64_t axis = innermost_rank; axis >= 0; --axis) {
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
      readIndex -= (int64_t)newIndex * srcStrides[axis];
      writeIndex -= (int64_t)newIndex * dstStrides[axis];
    }
  }
}

void __Wdma(uint64_t *src, uint64_t *dst, int *src_shape, int *src_stride,
            int *dst_shape, int *dst_stride, int rank, uint32_t elem_bytes,
            uint32_t fmt, uint32_t action, const void *base_dst,
            const char *kernel_name) {
  INTRNISIC_RUN_SWITCH;

  // Dynamic shape, kernel implementation will cause shape equal to 0
  for (int i = 0; i < rank; i++) {
    if (src_shape[i] == 0) {
      return;
    }
  }

  if (is_dma_action_logging(action)) {
    __EP_LOG__(3, "[func @%s] wdma -- src: %p, dst: %p, elem_bytes: %d (",
               kernel_name, src, dst, elem_bytes);

    __EP_LOG__(3, "src_shape[");
    for (int i = 0; i < rank; i++) {
      __EP_LOG__(3, "%d%s", src_shape[i], (i == rank - 1) ? "" : ", ");
    }
    __EP_LOG__(3, "], ");

    __EP_LOG__(3, "src_stride[");
    for (int i = 0; i < rank; i++) {
      __EP_LOG__(3, "%d%s", src_stride[i], (i == rank - 1) ? "" : ", ");
    }
    __EP_LOG__(3, "], ");

    __EP_LOG__(3, "dst_shape[");
    for (int i = 0; i < rank; i++) {
      __EP_LOG__(3, "%d%s", dst_shape[i], (i == rank - 1) ? "" : ", ");
    }
    __EP_LOG__(3, "], ");

    __EP_LOG__(3, "dst_stride[");
    for (int i = 0; i < rank; i++) {
      __EP_LOG__(3, "%d%s", dst_stride[i], (i == rank - 1) ? "" : ", ");
    }
    __EP_LOG__(3, "])\n");
  }

  if (is_dma_action_checking(action)) {
    uint64_t *header = get_header(base_dst);
    uint64_t CRC =
        crc32((uint8_t *)header, ClientPtrHeaderBytes - sizeof(uint64_t));
    if (CRC != header[2]) {
      __EP_LOG__(3,
                 "\n\nerror: [func @%s] wdma total_size %llu, magic %x, bad "
                 "crc %x != %x(header[2]), "
                 "base_dst (%p) data %x, src (%p)\n\n",
                 kernel_name, header[0], header[1], CRC, header[2], base_dst,
                 header[2], src);
    } else if (header[1] != ClientPtrMagic) {
      dma_bad_magic_count++;
      __EP_LOG__(
          3,
          "\n\nerror: [func @%s] wdma total_size %llu, bad magic %x, crc %x, "
          "base_dst (%p) data %x, dst (%p)\n\n",
          kernel_name, header[0], header[1], header[2], base_dst, header[2],
          dst);
    } else {
      uint64_t total_size = header[0];
      Tx81DstAddrRange range =
          compute_wdma_dst_addr_range(dst, src_shape, src_stride, dst_shape,
                                      dst_stride, rank, elem_bytes, fmt);

      __EP_LOG__(
          3,
          "[func @%s] wdma base_dst (%p), total_size %llu, magic %x, crc %x, "
          "dst (%p), rang_size %d\n",
          kernel_name, base_dst, total_size, header[1], header[2], dst,
          (unsigned int)(range.max_addr - range.min_addr));

      if (range.min_addr < (uintptr_t)base_dst ||
          range.max_addr > (uintptr_t)base_dst + total_size) {
        dma_oob_count++;
        __EP_LOG__(3,
                   "[func @%s] fatal error: ddr memory OOB, "
                   "wdma base_dst (%p), total_size %llu, dst (%p), "
                   "dma_oob_count (%d)\n",
                   kernel_name, base_dst, total_size, dst, dma_oob_count);
        if (get_dma_check_abort()) {
          abort();
        }
      }
    }
  }

  // If inner dim stride is 1, use scalar wdma.
  if (src_stride[rank - 1] != 1 || dst_stride[rank - 1] != 1) {
    __WdmaVectorize((char *)src, (char *)dst, src_shape, src_stride, dst_shape,
                    dst_stride, rank, elem_bytes, Fmt_INT8, rank - 1,
                    elem_bytes);
    return;
  }
  legalizeMemoryOpAttribute(src_shape, src_stride, dst_shape, dst_stride, rank,
                            &elem_bytes, &fmt);

  if (rank == 4 && no_reverse_memory_access(dst_stride, rank) &&
      is_contiguous(src_shape, src_stride, rank)) {
    __Wdma4d(dst, src, dst_shape[3], dst_stride[2], dst_shape[2], dst_stride[1],
             dst_shape[1], dst_stride[0], dst_shape[0], fmt);
    return;
  }

  __WdmaVectorize((char *)src, (char *)dst, src_shape, src_stride, dst_shape,
                  dst_stride, rank, elem_bytes, fmt, rank - 2,
                  src_shape[rank - 1]);
}
