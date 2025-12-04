//===------------------------ channelnorm.c -------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::channelnorm/dechannelnorm.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <stdio.h>

void __ChannelNorm(uint64_t *src, uint64_t *dst, uint16_t n, uint16_t h,
                   uint16_t w, uint16_t c, uint16_t c0, uint16_t bit_width) {
  int calign_base = bit_width == 8 ? 128 : 64;
  int dtype_size = bit_width / 8;
  int cx = c / calign_base;

  TsmDataMove *dm = g_intrinsic()->datamove_pointer;
  TsmDataMoveInstr dm_param = {I_CGRA,
                               {
                                   0,
                               },
                               {
                                   0,
                               }};

  uint32_t inner_dim_size = c;
  St_StrideIteration src_it = {0}, dst_it = {0};

  // align cx
  if (cx > 0) {
    uint32_t elem_size = calign_base * dtype_size;
    // byte number
    src_it.stride0 = inner_dim_size * dtype_size;
    src_it.iteration0 = h * w;
    src_it.stride1 = elem_size;
    src_it.iteration1 = cx;
    src_it.stride2 = h * w * c * dtype_size;
    src_it.iteration2 = n;

    dst_it.stride0 = elem_size;
    dst_it.iteration0 = h * w;
    dst_it.stride1 = dst_it.iteration0 * dst_it.stride0;
    dst_it.iteration1 = cx;
    dst_it.stride2 = n * h * w * elem_size;
    dst_it.iteration2 = n;

    dm->GatherScatter(&dm_param, (uint64_t)src, (uint64_t)dst, elem_size,
                      &src_it, &dst_it);
    TsmExecute(&dm_param);
    TsmWaitfinish();
  }

  // align c0
  if (c0 > 0) {
    uint32_t src_offset = cx * calign_base * dtype_size;
    uint32_t dst_offset = cx * h * w * calign_base * dtype_size;
    int32_t c0_valid = inner_dim_size - cx * calign_base;
    int32_t elem_size = c0_valid * dtype_size;

    src_it.stride0 = inner_dim_size * dtype_size;
    src_it.iteration0 = n * h * w;
    src_it.stride1 = n * h * w * inner_dim_size * dtype_size;
    src_it.iteration1 = 1;
    src_it.stride2 = n * h * w * inner_dim_size * dtype_size;
    src_it.iteration2 = 1;

    dst_it.stride0 = c0 * dtype_size;
    dst_it.iteration0 = n * h * w;
    dst_it.stride1 = n * h * w * c0 * dtype_size;
    dst_it.iteration1 = 1;
    dst_it.stride2 = n * h * w * c0 * dtype_size;
    dst_it.iteration2 = 1;

    dm->GatherScatter(&dm_param, (uint64_t)src + src_offset,
                      (uint64_t)dst + dst_offset, elem_size, &src_it, &dst_it);
    TsmExecute(&dm_param);
    TsmWaitfinish();
  }
}

void __DechannelNorm(uint64_t *src, uint64_t *dst, uint16_t n, uint16_t h,
                     uint16_t w, uint16_t c, uint16_t c0, uint16_t bit_width) {
  int calign_base = bit_width == 8 ? 128 : 64;
  int dtype_size = bit_width / 8;
  int cx = c / calign_base;

  TsmDataMove *dm = g_intrinsic()->datamove_pointer;
  TsmDataMoveInstr dm_param = {I_CGRA,
                               {
                                   0,
                               },
                               {
                                   0,
                               }};

  uint32_t inner_dim_size = c;
  St_StrideIteration src_it = {0}, dst_it = {0};

  // align cx
  if (cx > 0) {
    uint32_t elem_size = calign_base * dtype_size;
    // byte number

    src_it.stride0 = h * w * elem_size;
    src_it.iteration0 = cx;
    src_it.stride1 = elem_size;
    src_it.iteration1 = h * w;
    src_it.stride2 = cx * h * w * elem_size;
    src_it.iteration2 = n;

    dst_it.stride0 = elem_size;
    dst_it.iteration0 = cx;
    dst_it.stride1 = inner_dim_size * dtype_size;
    dst_it.iteration1 = h * w;
    dst_it.stride2 = h * w * inner_dim_size * dtype_size;
    dst_it.iteration2 = n;

    dm->GatherScatter(&dm_param, (uint64_t)src, (uint64_t)dst, elem_size,
                      &src_it, &dst_it);
    TsmExecute(&dm_param);
    TsmWaitfinish();
  }

  // align c0
  if (c0 > 0) {
    uint32_t src_offset = cx * calign_base * dtype_size;
    uint32_t dst_offset = cx * h * w * calign_base * dtype_size;
    int32_t c0_valid = inner_dim_size - cx * calign_base;
    int32_t elem_size = c0_valid * dtype_size;

    src_it.stride0 = c0 * dtype_size;
    src_it.iteration0 = n * h * w;
    src_it.stride1 = n * h * w * c0 * dtype_size;
    src_it.iteration1 = 1;
    src_it.stride2 = n * h * w * c0 * dtype_size;
    src_it.iteration2 = 1;

    dst_it.stride0 = inner_dim_size * dtype_size;
    dst_it.iteration0 = n * h * w;
    dst_it.stride1 = n * h * w * inner_dim_size * dtype_size;
    dst_it.iteration1 = 1;
    dst_it.stride2 = n * h * w * inner_dim_size * dtype_size;
    dst_it.iteration2 = 1;

    dm->GatherScatter(&dm_param, (uint64_t)src + src_offset,
                      (uint64_t)dst + dst_offset, elem_size, &src_it, &dst_it);
    TsmExecute(&dm_param);
    TsmWaitfinish();
  }
}
