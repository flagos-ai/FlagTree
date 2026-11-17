// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIV/Vector) into its own bitcode.
#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

// Dequantise rows of a paged FP8 KV cache into a BF16 tile in GM.
//
// The record format is fixed: 448 e4m3 values, then 64 BF16 values that are
// already in their final form (a RoPE tail, copied through untouched), for
// 512 output columns per row. A page holds `page_slots` such records of
// `record_bytes` each, followed at `scale_offset` by eight bytes per token:
// seven ue8m0 group scales and one pad byte, one scale per 64 e4m3 values.
// Only the page geometry is parameterised, so one operator serves any paged
// cache that packs this record.
//
// Two properties of that format make this worth an operator rather than
// elementwise Triton:
//
//  * e4m3 -> bf16 is a bit move, not arithmetic. `vconv_s82f16` does the
//    widening in hardware and takes 256 bytes per issue, where 16-bit
//    elementwise ops take 128 lanes; the whole tile then needs about two
//    dozen instructions, each carrying `repeat = rows`. The same transform
//    written as Triton elementwise ops costs about 26 instructions *per row*
//    -- the row loop cannot be handed to the sequencer -- and measured 6.5x
//    the vector-pipe time at equal element counts.
//
//  * applying a ue8m0 scale is exact in the exponent field: the scale is
//    2^(s-127) and a bit-moved e4m3 byte is 2^(e-127)*(1+m/8), so adding
//    s-7 to the exponent is the whole multiply. `vbrcb` broadcasts the eight
//    packed scale words of a row into lanes and a strided `vadd` applies
//    them without materialising a scale tile.
//
// The caller passes the row map rather than resolved addresses because the
// run length of each physical run is folded into its slot's high bits
// (`run = location / page_slots`): one run becomes one DMA burst, so a
// sixteen-row window tile issues one copy instead of sixteen. Resolving the
// map outside the op would lose that.
//
// Rows at or past `count` are zeroed. `win_count` splits the map into a
// leading window region and a trailing region that indexes the second pool,
// which is how a sliding window and a compressed selection share one tile.
extern "C" {
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_dequant_e4m3_ue8m0_rows_bf16(
    memref_t<__gm__ int64_t, 2> *win_pool,
    memref_t<__gm__ int64_t, 2> *cmp_pool, memref_t<__gm__ int32_t, 2> *pages,
    memref_t<__gm__ int32_t, 2> *slots, int64_t first_row, int64_t win_count,
    int64_t count, int64_t page_stride, int64_t record_bytes,
    int64_t scale_offset, int64_t page_slots, int64_t window_extent,
    memref_t<__gm__ bfloat16_t, 2> *dst,
    memref_t<__ubuf__ bfloat16_t, 2> *scratch) {
  auto wp = win_pool->aligned + win_pool->offset;
  auto cp = cmp_pool->aligned + cmp_pool->offset;
  auto pp = pages->aligned + pages->offset;
  auto sp = slots->aligned + slots->offset;
  const int64_t rows = scratch->sizes[0];
  // Scratch layout, in 16-bit columns per row: the 512 output values, then 16
  // columns of scale slots, then the token's packed record. The total is the
  // width the Python side declares as the op's scratch.
  auto final = (__ubuf__ uint16_t *)(scratch->aligned + scratch->offset);
  auto scales = final + rows * 512;
  auto raw = (__ubuf__ uint8_t *)(final + rows * 528);
  const int64_t valid_rows = min(rows, max((int64_t)0, count - first_row));
  INTRINSIC(set_mask_norm);
  INTRINSIC(set_vector_mask, 0, (uint64_t)-1);
  if (valid_rows < rows) {
    INTRINSIC(set_mask_count);
    INTRINSIC(set_vector_mask, 0, (rows - valid_rows) * record_bytes / 2);
    INTRINSIC(vector_dup, (__ubuf__ half *)(raw + valid_rows * record_bytes),
              (half)0, (uint8_t)1, (uint16_t)1, 1, (uint8_t)8, 0);
  }
  INTRINSIC(set_flag, PIPE_V, PIPE_MTE2, EVENT_ID7);
  INTRINSIC(wait_flag, PIPE_V, PIPE_MTE2, EVENT_ID7);
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t row = first_row + i;
    if (row >= count)
      continue;
    const bool window = row < win_count;
    const int64_t logical = window ? row : window_extent + row - win_count;
    const int64_t page = pp[logical * pages->strides[0]];
    const int64_t location = sp[logical * slots->strides[0]];
    const int64_t slot = location % page_slots;
    auto pool = (__gm__ uint8_t *)(window ? wp : cp);
    auto src = pool + page * page_stride + slot * record_bytes;
    const int64_t run =
        min(rows - i, min(location / page_slots,
                          (window ? min(win_count, count) : count) - row));
    INTRINSIC(copy_gm_to_ubuf_align_b8, raw + i * record_bytes, src, 0, 1,
              run * record_bytes, 0, 0, 0, 0);
    INTRINSIC(copy_gm_to_ubuf_align_b8, (__ubuf__ uint8_t *)(scales + i * 16),
              pool + page * page_stride + scale_offset + slot * 8, 0, run, 8, 0,
              0, 0, 0);
    i += run - 1;
  }
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, EVENT_ID7);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, EVENT_ID7);
  // The previous call's store has to drain before its scratch is overwritten.
  INTRINSIC(set_flag, PIPE_MTE3, PIPE_V, EVENT_ID5);
  INTRINSIC(wait_flag, PIPE_MTE3, PIPE_V, EVENT_ID5);
  INTRINSIC(set_mask_norm);
  INTRINSIC(set_vector_mask, (uint64_t)-1, (uint64_t)-1);
  for (int64_t group = 0; group < 4; ++group) {
    INTRINSIC(vconv_s82f16, (__ubuf__ half *)(final + group * 128),
              (__ubuf__ int8_t *)(raw + group * 128), (uint8_t)rows,
              (uint16_t)1, (uint16_t)1, (uint16_t)32, (uint16_t)18);
  }
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(set_mask_count);
  INTRINSIC(set_vector_mask, 0, rows * 512);
  INTRINSIC(vconv_f162s16r, (__ubuf__ int16_t *) final, (__ubuf__ half *) final,
            (uint8_t)1, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vshl, final, final, (uint16_t)4, (uint8_t)1, (uint16_t)1,
            (uint16_t)1, (uint16_t)8, (uint16_t)8);
  INTRINSIC(pipe_barrier, PIPE_V);
  // The e4m3 bytes are dead after conversion. Reuse their first 256 bytes for
  // the mask, keeping the BF16 tail intact. 0x87f0 keeps a BF16's sign and
  // exponent and clears the mantissa bits the shift above moved in.
  INTRINSIC(set_vector_mask, 0, 128);
  INTRINSIC(vector_dup, (__ubuf__ int16_t *)raw, (int16_t)0x87f0, (uint8_t)1,
            (uint16_t)1, 1, (uint8_t)8, 0);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(set_vector_mask, 0, rows * 512);
  INTRINSIC(vand, final, final, (__ubuf__ uint16_t *)raw, (uint8_t)1,
            (uint16_t)1, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8,
            (uint16_t)0);
  INTRINSIC(pipe_barrier, PIPE_V);
  // Preserve the BF16 tail before reusing the whole raw span for the biases.
  INTRINSIC(copy_ubuf_to_ubuf, (__ubuf__ uint8_t *)(final + 448), raw + 448, 0,
            rows, 4, 14, 28);
  INTRINSIC(pipe_barrier, PIPE_V);
  auto bias = (__ubuf__ uint16_t *)raw;
  auto temp = bias + rows * 256;
  // A ue8m0 byte s scales by 2**(s-127) and e4m3's own bias is 7, so biasing
  // the BF16 exponent field by (s - 7) << 7 applies the scale exactly. Four
  // uint16 words hold all eight scale bytes of each 32-byte slot, so one
  // reduction compacts the tile and the arithmetic then runs once.
  INTRINSIC(set_mask_norm);
  INTRINSIC(set_vector_mask, 0, 255);
  INTRINSIC(vector_dup, (__ubuf__ int16_t *)bias, (int16_t)0x000f, (uint8_t)1,
            (uint16_t)1, 1, (uint8_t)8, 0);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(set_vector_mask, (uint64_t)-1, (uint64_t)-1);
  INTRINSIC(vreducev2, temp, scales, bias, (uint8_t)(rows / 8), (uint16_t)1,
            (uint8_t)0, (uint16_t)8, (uint16_t)0);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(set_mask_count);
  INTRINSIC(set_vector_mask, 0, rows * 8);
  INTRINSIC(vconv_u82f16, (__ubuf__ half *)scales, (__ubuf__ uint8_t *)temp,
            (uint8_t)1, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vconv_f162s16r, (__ubuf__ int16_t *)scales, (__ubuf__ half *)scales,
            (uint8_t)1, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vadds, (__ubuf__ int16_t *)scales, (__ubuf__ int16_t *)scales,
            (int16_t)-7, (uint8_t)1, (uint16_t)1, (uint16_t)1, (uint16_t)8,
            (uint16_t)8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vshl, scales, scales, (uint16_t)7, (uint8_t)1, (uint16_t)1,
            (uint16_t)1, (uint16_t)8, (uint16_t)8);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(set_mask_norm);
  if (valid_rows > 0) {
    // Eight packed scale words per row become eight 16-lane blocks. Each
    // group then broadcasts one block over its 64 lanes, across all rows.
    INTRINSIC(set_vector_mask, 0, (uint64_t)-1);
    INTRINSIC(vbrcb, bias, scales, (uint16_t)1, (uint16_t)8,
              (uint8_t)valid_rows);
    INTRINSIC(pipe_barrier, PIPE_V);
    for (int64_t group = 0; group < 7; ++group) {
      auto target = (__ubuf__ int16_t *)(final + group * 64);
      INTRINSIC(vadd, target, target, (__ubuf__ int16_t *)(bias + group * 16),
                (uint8_t)valid_rows, (uint16_t)1, (uint16_t)1, (uint16_t)0,
                (uint16_t)32, (uint16_t)32, (uint16_t)8);
    }
    INTRINSIC(pipe_barrier, PIPE_V);
  }
  INTRINSIC(set_mask_norm);
  INTRINSIC(set_vector_mask, (uint64_t)-1, (uint64_t)-1);
  INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, EVENT_ID5);
  INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, EVENT_ID5);
  INTRINSIC(copy_ubuf_to_gm_align_b8,
            (__gm__ uint8_t *)(dst->aligned + dst->offset),
            (__ubuf__ uint8_t *) final, 0, 1, rows * 512 * 2, 0, 0, 0, 0);
}
}
