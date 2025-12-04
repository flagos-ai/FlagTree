//===------------------------ wdma.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Wdma, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <stdio.h>

// Only support rank = 4.
void __Wdma4d(uint64_t *src, uint64_t *dst, int *src_shape, int *src_stride,
              int *dst_shape, int *dst_stride, int rank, uint32_t elem_bytes,
              uint32_t fmt) {
  TsmWdma *wdma = g_intrinsic()->wdma_pointer;
  TsmWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  wdma->AddSrcDst(&inst, (uint64_t)src, (uint64_t)dst, (Data_Format)fmt);
  wdma->ConfigStrideIteration(&inst, dst_shape[3], dst_stride[2], dst_shape[2],
                              dst_stride[1], dst_shape[1], dst_stride[0],
                              dst_shape[0]);
  TsmExecute(&inst);
  TsmWaitfinish();
}

// Wdma line by line.
void __WdmaVectorize(char *srcPtr, char *dstPtr, int *src_shape,
                     int *src_stride, int *dst_shape, int *dst_stride, int rank,
                     uint32_t elem_bytes, uint32_t fmt, int innermost_rank,
                     int inner_elem_count) {
  TsmWdma *wdma = g_intrinsic()->wdma_pointer;
  TsmWdmaInstr inst = {I_WDMA,
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
    srcStrides[rankp] = src_stride[rankp] * elem_bytes;
    dstStrides[rankp] = dst_stride[rankp] * elem_bytes;
  }

  for (;;) {
    // Copy inner dim, line by line.
    wdma->Wdma1d(&inst, (uint64_t)(srcPtr + readIndex),
                 (uint64_t)(dstPtr + writeIndex), inner_elem_count,
                 (Data_Format)fmt);
    TsmExecute(&inst);
    TsmWaitfinish();

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
      readIndex -= src_shape[axis] * srcStrides[axis];
      writeIndex -= dst_shape[axis] * dstStrides[axis];
    }
  }
}

void __Wdma(uint64_t *src, uint64_t *dst, int *src_shape, int *src_stride,
            int *dst_shape, int *dst_stride, int rank, uint32_t elem_bytes,
            uint32_t fmt) {

  // Dynamic shape, kernel implementation will cause shape equal to 0
  for (int i = 0; i < rank; i++) {
    if (src_shape[i] == 0) {
      return;
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

  if (rank == 4 && is_contiguous(dst_shape, dst_stride, elem_bytes)) {
    __Wdma4d(src, dst, src_shape, src_stride, dst_shape, dst_stride, rank,
             elem_bytes, fmt);
    return;
  }

  __WdmaVectorize((char *)src, (char *)dst, src_shape, src_stride, dst_shape,
                  dst_stride, rank, elem_bytes, fmt, rank - 2,
                  src_shape[rank - 1]);
}
