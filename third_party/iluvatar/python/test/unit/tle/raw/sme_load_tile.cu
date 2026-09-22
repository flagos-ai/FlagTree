#include <stdint.h>

// Drives the Streaming Memory Engine directly from a raw device function.
// __ivcorex_sme_load_16x1b64_imm and __ivcorex_sl_waitcnt are corex clang
// builtins with no NVIDIA CUDA counterpart: the first issues
// sl_sme_load_16x1b64, one 16 row x 64 byte hardware tile, and the second
// drains the G2S queue (bit 3 enables G2S, G2S_CNT=0 waits for all of them).
//
// Intrinsic arguments are byte counts. The global row stride and the column
// offset have to be 64B aligned, the shared pointer 4B aligned, and the last
// three arguments must be immediates.
#define SME_TILE_ROWS 16

__device__ auto
SmeLoadTiles(__attribute__((address_space(3))) float *out_allocated,
             __attribute__((address_space(3))) float *out_aligned,
             const int64_t out_offset, const int64_t out_size0,
             const int64_t out_size1, const int64_t out_stride0,
             const int64_t out_stride1,
             __attribute__((address_space(1))) const float *src,
             const int32_t stride_bytes) {
  // The engine is warp-uniform, so every lane issues the same transfer; one
  // hardware tile covers 16 rows and the caller sizes the buffer in multiples
  // of that.
  __attribute__((address_space(3))) float *dst = out_aligned + out_offset;
  for (int64_t tile = 0; tile < out_size0 / SME_TILE_ROWS; ++tile) {
    __ivcorex_sme_load_16x1b64_imm(
        (void *)(dst + tile * SME_TILE_ROWS * out_size1), (const void *)src,
        (unsigned)stride_bytes, (unsigned)(tile * SME_TILE_ROWS * stride_bytes),
        /*RowOffsetImm=*/0u, /*ColOffsetImm=*/0u,
        /*ShareMemoryOffsetImm=*/0u);
  }
  __ivcorex_sl_waitcnt(8); // SME G2S
  __syncthreads();

  struct {
    __attribute__((address_space(3))) float *allocated;
    __attribute__((address_space(3))) float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  } r{out_allocated,
      out_aligned,
      out_offset,
      {out_size0, out_size1},
      {out_stride0, out_stride1}};
  return r;
}
