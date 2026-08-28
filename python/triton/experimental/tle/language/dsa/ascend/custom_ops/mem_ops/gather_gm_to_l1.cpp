// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIC/Cube) into its own bitcode.
#include "Utils.h"

#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

constexpr int32_t INTR_BYTES_PER_BLOCK = 32;

template <typename T> struct nd2nz_intrin_args {
  __cbuf__ T *dst_ptr;
  __gm__ T *src_ptr;
  uint8_t sid;
  uint16_t ndNum;
  uint16_t nValue;
  uint16_t dValue;
  uint16_t srcNdMatrixStride;
  uint16_t srcDValue;
  uint16_t dstNzC0Stride;
  uint16_t dstNzNStride;
  uint16_t dstNzMatrixStride;
};

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T> args) {
  if constexpr (sizeof(T) == 2) {
    INTRINSIC(copy_gm_to_cbuf_multi_nd2nz_b16, args.dst_ptr, args.src_ptr,
              args.sid, args.ndNum, args.nValue, args.dValue,
              args.srcNdMatrixStride, args.srcDValue, args.dstNzC0Stride,
              args.dstNzNStride, args.dstNzMatrixStride);
  } else if constexpr (sizeof(T) == 4) {
    INTRINSIC(copy_gm_to_cbuf_multi_nd2nz_b32s, args.dst_ptr, args.src_ptr,
              args.sid, args.ndNum, args.nValue, args.dValue,
              args.srcNdMatrixStride, args.srcDValue, args.dstNzC0Stride,
              args.dstNzNStride, args.dstNzMatrixStride);
  }
}

// Generic row gather: out row i <- src row index[i]. Both src rows and the
// index array live in GM and are assumed row-contiguous. Consecutive index
// rows (index[i + 1] == index[i] + 1) are coalesced into one 2-row ND2NZ
// copy.
template <typename T>
__aicore__ __attribute__((always_inline)) void
gather_gm_to_l1_impl(memref_t<__gm__ T, 2> *src,
                     memref_t<__gm__ int32_t, 2> *index, int64_t tile_size,
                     int64_t cols, memref_t<__cbuf__ T, 4> *dst) {

  auto gm_src = src->aligned + src->offset;
  auto gm_idx = index->aligned + index->offset;
  auto l1_ptr = dst->aligned + dst->offset;
  int64_t n_tile_ceil = dst->strides[0] / dst->strides[2];
  int64_t idx_stride = index->strides[0];
  int64_t c0_size = INTR_BYTES_PER_BLOCK / sizeof(T);

  for (int64_t i = 0; i < tile_size;) {
    int64_t row = static_cast<int64_t>(gm_idx[i * idx_stride]);

    __gm__ T *src_row = gm_src + row * cols;
    __cbuf__ T *dst_row = l1_ptr + i * c0_size;

    if (i + 1 < tile_size &&
        static_cast<int64_t>(gm_idx[(i + 1) * idx_stride]) == row + 1) {
      copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
          dst_row, src_row, 0, 1, 2, static_cast<uint16_t>(cols), 0,
          static_cast<uint16_t>(cols), static_cast<uint16_t>(n_tile_ceil), 1,
          1});

      i += 2;
      continue;
    }

    copy_gm_to_cbuf_intrin_core(nd2nz_intrin_args<T>{
        dst_row, src_row, 0, 1, 1, static_cast<uint16_t>(cols), 0, 0,
        static_cast<uint16_t>(n_tile_ceil), 0, 1});

    i += 1;
  }
}

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_gm_to_l1_half(memref_t<__gm__ half, 2> *src,
                                         memref_t<__gm__ int32_t, 2> *index,
                                         int64_t tile_size, int64_t cols,
                                         memref_t<__cbuf__ half, 4> *dst) {
  gather_gm_to_l1_impl<half>(src, index, tile_size, cols, dst);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_gm_to_l1_bf16(
    memref_t<__gm__ bfloat16_t, 2> *src, memref_t<__gm__ int32_t, 2> *index,
    int64_t tile_size, int64_t cols, memref_t<__cbuf__ bfloat16_t, 4> *dst) {
  gather_gm_to_l1_impl<bfloat16_t>(src, index, tile_size, cols, dst);
}
}
