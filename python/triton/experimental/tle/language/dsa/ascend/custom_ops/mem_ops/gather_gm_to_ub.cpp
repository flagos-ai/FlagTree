// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIV/Vector) into its own bitcode.
#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_gm_to_ub_b16(__gm__ T *gm_ptr, __ubuf__ T *ub_ptr, uint16_t burst_cnt,
                  uint32_t burst_len, uint32_t src_gap, uint32_t dst_gap) {
  INTRINSIC(copy_gm_to_ubuf_align_b16, ub_ptr, gm_ptr, 0, burst_cnt, burst_len,
            0, 0, src_gap, dst_gap);
}

// Generic row gather: out row i <- src row index[i]. Both src rows and the
// index array live in GM and are assumed row-contiguous. Consecutive index
// rows (index[i + 1] == index[i] + 1) are coalesced into one 2-burst copy.
template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_gm_to_ub_impl(memref_t<__gm__ T, 2> *src,
                     memref_t<__gm__ int32_t, 2> *index, int64_t tile_size,
                     int64_t cols, memref_t<__ubuf__ T, 2> *dst) {
  auto gm_src = src->aligned + src->offset;
  auto gm_idx = index->aligned + index->offset;
  auto ub_dst = dst->aligned + dst->offset;

  uint32_t row_bytes = static_cast<uint32_t>(cols * sizeof(T));
  int64_t idx_stride = index->strides[0];
  int64_t dst_stride = dst->strides[0];
  uint32_t dst_gap = static_cast<uint32_t>((dst_stride - cols) * sizeof(T));

  for (int64_t i = 0; i < tile_size;) {
    int64_t row = static_cast<int64_t>(gm_idx[i * idx_stride]);

    __gm__ T *src_row = gm_src + row * cols;
    __ubuf__ T *dst_row = ub_dst + i * dst_stride;

    if (i + 1 < tile_size &&
        static_cast<int64_t>(gm_idx[(i + 1) * idx_stride]) == row + 1) {
      copy_gm_to_ub_b16(src_row, dst_row, 2, row_bytes, 0, dst_gap);
      i += 2;
      continue;
    }

    copy_gm_to_ub_b16(src_row, dst_row, 1, row_bytes, 0, 0);
    i += 1;
  }
}

extern "C" {

__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_gm_to_ub_half(memref_t<__gm__ half, 2> *src,
                                         memref_t<__gm__ int32_t, 2> *index,
                                         int64_t tile_size, int64_t cols,
                                         memref_t<__ubuf__ half, 2> *dst) {
  gather_gm_to_ub_impl<half>(src, index, tile_size, cols, dst);
}

__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_gm_to_ub_bf16(
    memref_t<__gm__ bfloat16_t, 2> *src, memref_t<__gm__ int32_t, 2> *index,
    int64_t tile_size, int64_t cols, memref_t<__ubuf__ bfloat16_t, 2> *dst) {
  gather_gm_to_ub_impl<bfloat16_t>(src, index, tile_size, cols, dst);
}
}
