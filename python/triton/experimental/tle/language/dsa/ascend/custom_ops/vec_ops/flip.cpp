// Copyright 2026 FlagOS Contributors
// Copyright 2026- XCoreSigma Technology Co.

#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

// Vectorized in-UB reversal along the last dim of a 2D tile, replacing the
// per-element scalar loop of the meta-op library Flip1D.cpp.
//
// Building blocks: the NCHW transpose family scatter_vnchwconv_{b8, b16};
// the 16 src / 16 dst row addresses are set through the VA registers and
// one repeat computes D[delta(i)][j] = S[sigma(j)][i] (measured on 910B3),
// so a single call can never reverse; two calls through a 512B scratch
// compose it:
//   call1: T[a][b] = S[15 - b][a]          (src -> scratch)
//   call2: D[m][n] = T[sigma2(n)][m]       (scratch -> dst)
// - b16 (2-byte dtypes): sigma2(i) = 15 - i.
// - b16 word_mode (4-byte dtypes): a word is two adjacent halves, so the
//   halves must travel in pairs, sigma2(i) = 15 - (i ^ 1).
// - b8 (1-byte dtypes): a row is the low or high 16B half of a 32B unit
//   (dstHighHalf/srcHighHalf template operands); the chunk holds two
//   interleaved 16x16 byte matrices, each transposed into the matching
//   scratch half and transposed back with the half roles swapped.
// Single-repeat calls use stride 0, back-to-back calls are separated by a
// pipe_barrier, and chunks are visited in reverse at the loop level (the
// repeat engine only walks addresses forward).
//
// Coverage: row bytes multiple of 512 for 1-/2-/4-byte dtypes; 8-byte
// dtypes and shorter rows fall back to the library scalar loop. tmp
// provides one 512B 32B-aligned scratch. No other barrier is added;
// ordering versus neighbouring ops is the caller's responsibility.
namespace {

constexpr uint32_t FLIP_BLK = 32;    // one 32B addressable row unit
constexpr uint32_t B8_CHUNK = 512;   // two interleaved 16x16 byte matrices
constexpr uint32_t B16_CHUNK = 512;  // 16x16 halves per chunk
constexpr uint32_t MAX_REPEATS = 255;

__aiv__ __attribute__((always_inline)) uint32_t
src_row_of(uint32_t i, bool word_mode) {
  return word_mode ? (15u - (i ^ 1u)) : (15u - i);
}

// One b16 transpose call over a single 512B chunk: src rows fed through
// `src_row_of`, dst rows in forward order.
__aiv__ __attribute__((always_inline)) void
transpose_chunk_b16(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src,
                    bool word_mode) {
  uint64_t src_lo[8], src_hi[8], dst_lo[8], dst_hi[8];
  for (uint32_t i = 0; i < 8; ++i) {
    src_lo[i] = (uint64_t)(src + src_row_of(i, word_mode) * FLIP_BLK);
    src_hi[i] = (uint64_t)(src + src_row_of(8 + i, word_mode) * FLIP_BLK);
    dst_lo[i] = (uint64_t)(dst + i * FLIP_BLK);
    dst_hi[i] = (uint64_t)(dst + (8 + i) * FLIP_BLK);
  }
  set_va_reg_sb(VA2, src_lo);
  set_va_reg_sb(VA3, src_hi);
  set_va_reg_sb(VA0, dst_lo);
  set_va_reg_sb(VA1, dst_hi);
  scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
}

// One b8 transpose call over one 16x16 byte matrix: each row is the low or
// high 16B half of a 32B unit (a compile-time bool per the intrinsic ABI),
// 16 units at FLIP_BLK stride; src rows reversed, dst rows forward.
template <bool DstHigh, bool SrcHigh>
__aiv__ __attribute__((always_inline)) void
transpose_chunk_b8(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src) {
  uint64_t src_lo[8], src_hi[8], dst_lo[8], dst_hi[8];
  for (uint32_t i = 0; i < 8; ++i) {
    src_lo[i] = (uint64_t)(src + (15u - i) * FLIP_BLK);
    src_hi[i] = (uint64_t)(src + (7u - i) * FLIP_BLK);
    dst_lo[i] = (uint64_t)(dst + i * FLIP_BLK);
    dst_hi[i] = (uint64_t)(dst + (8u + i) * FLIP_BLK);
  }
  set_va_reg_sb(VA2, src_lo);
  set_va_reg_sb(VA3, src_hi);
  set_va_reg_sb(VA0, dst_lo);
  set_va_reg_sb(VA1, dst_hi);
  scatter_vnchwconv_b8(VA0, VA2, 1, 0, 0, DstHigh, SrcHigh);
}

// Same element-by-element loop as the library flip_1d, including its
// V<->S barriers; only used for shapes the transpose path cannot cover.
template <typename T>
__aiv__ __attribute__((always_inline)) void
reverse_scalar(__ubuf__ T *dst, __ubuf__ T *src, int64_t size0,
               int64_t stride0) {
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < size0; ++i) {
    *(dst + stride0 * i) = *(src + stride0 * (size0 - 1 - i));
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

// Two-call b16 reversal of one 512B chunk: transpose into the scratch,
// transpose back through the sigma2 table.
__aiv__ __attribute__((always_inline)) void
reverse_chunk_b16(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *tmp,
                  __ubuf__ uint8_t *src, bool word_mode) {
  transpose_chunk_b16(tmp, src, false);
  pipe_barrier(PIPE_V);
  transpose_chunk_b16(dst, tmp, word_mode);
  pipe_barrier(PIPE_V);
}

// Four-call b8 reversal of one 512B chunk: the two interleaved byte
// matrices are transposed into the matching scratch halves, then
// transposed back with the half roles swapped, which composes the
// byte-wise reversal D(u,s,j) = S(15-u, 1-s, 15-j).
__aiv__ __attribute__((always_inline)) void
reverse_chunk_b8(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *tmp,
                 __ubuf__ uint8_t *src) {
  transpose_chunk_b8<false, false>(tmp, src);
  pipe_barrier(PIPE_V);
  transpose_chunk_b8<true, true>(tmp, src);
  pipe_barrier(PIPE_V);
  transpose_chunk_b8<false, true>(dst, tmp);
  pipe_barrier(PIPE_V);
  transpose_chunk_b8<true, false>(dst, tmp);
  pipe_barrier(PIPE_V);
}

// Reversal of every 512B chunk of one region of `row_cnt` contiguous rows,
// chunks iterated in reverse within each row through the fixed scratch.
// Returns false when the rows do not cover whole chunks, leaving the data
// untouched for the scalar fallback.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
reverse_chunked(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *tmp,
                __ubuf__ uint8_t *src, uint32_t row_bytes,
                uint32_t row_cnt) {
  if (row_bytes % B16_CHUNK != 0) {
    return false;
  }
  const uint32_t chunks = row_bytes / B16_CHUNK;
  for (uint32_t r = 0; r < row_cnt; ++r) {
    __ubuf__ uint8_t *srow = src + (uint64_t)r * row_bytes;
    __ubuf__ uint8_t *drow = dst + (uint64_t)r * row_bytes;
    for (uint32_t c = 0; c < chunks; ++c) {
      __ubuf__ uint8_t *sc = srow + (chunks - 1 - c) * B16_CHUNK;
      __ubuf__ uint8_t *dc = drow + c * B16_CHUNK;
      if constexpr (sizeof(T) == 1) {
        reverse_chunk_b8(dc, tmp, sc);
      } else if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
        reverse_chunk_b16(dc, tmp, sc, sizeof(T) == 4);
      } else {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
flip_impl(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *tmp,
          memref_t<__ubuf__ T, 2> *dst) {
  const int64_t rows = src->sizes[0];
  const int64_t cols = src->sizes[1];
  __ubuf__ uint8_t *sp =
      reinterpret_cast<__ubuf__ uint8_t *>(src->aligned + src->offset);
  __ubuf__ uint8_t *tp =
      reinterpret_cast<__ubuf__ uint8_t *>(tmp->aligned + tmp->offset);
  __ubuf__ uint8_t *dp =
      reinterpret_cast<__ubuf__ uint8_t *>(dst->aligned + dst->offset);
  const uint32_t row_bytes = (uint32_t)(cols * sizeof(T));
  const bool blk_aligned =
      (((uint64_t)sp | (uint64_t)tp | (uint64_t)dp) % FLIP_BLK) == 0;
  const bool contiguous =
      blk_aligned && src->strides[0] == cols && dst->strides[0] == cols &&
      src->strides[1] == 1 && dst->strides[1] == 1;

  if (contiguous && reverse_chunked<T>(dp, tp, sp, row_bytes, (uint32_t)rows)) {
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    __ubuf__ uint8_t *srow = sp + (uint64_t)r * src->strides[0] * sizeof(T);
    __ubuf__ uint8_t *drow = dp + (uint64_t)r * dst->strides[0] * sizeof(T);
    if (blk_aligned && src->strides[1] == 1 &&
        reverse_chunked<T>(drow, tp, srow, row_bytes, 1)) {
      continue;
    }
    reverse_scalar<T>(dst->aligned + dst->offset + r * dst->strides[0],
                      src->aligned + src->offset + r * src->strides[0], cols,
                      src->strides[1]);
  }
}

} // namespace

#define REGISTER_CUSTOM_FLIP(TYPE_NAME, T)                                      \
  extern "C" __aiv__ __attribute__((always_inline)) void                        \
      _mlir_ciface_custom_flip_##TYPE_NAME(                                     \
          memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *tmp,           \
          memref_t<__ubuf__ T, 2> *dst) {                                       \
    flip_impl<T>(src, tmp, dst);                                                \
  }

REGISTER_CUSTOM_FLIP(half, half)
REGISTER_CUSTOM_FLIP(bf16, bfloat16_t)
REGISTER_CUSTOM_FLIP(float, float)
REGISTER_CUSTOM_FLIP(short, int16_t)
REGISTER_CUSTOM_FLIP(ushort, uint16_t)
REGISTER_CUSTOM_FLIP(int, int32_t)
REGISTER_CUSTOM_FLIP(uint, uint32_t)
REGISTER_CUSTOM_FLIP(char, int8_t)
REGISTER_CUSTOM_FLIP(uchar, uint8_t)
REGISTER_CUSTOM_FLIP(long, int64_t)
REGISTER_CUSTOM_FLIP(ulong, uint64_t)
