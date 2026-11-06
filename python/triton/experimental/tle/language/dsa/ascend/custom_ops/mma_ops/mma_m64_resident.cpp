// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIC/Cube) into its own bitcode.
#include "mma_common.h"

// M64 MMA that keeps `persistent_cols` output columns resident in L0C across
// a queue of tiles, so a running sum is never written to GM and read back.
//
// `kind` 0 multiplies a [64, 512] A tile by a [bn, 512] B tile transposed;
// `kind` 1 multiplies a [64, bn] A tile by the resident B tile, accumulating
// the first `persistent_cols` columns into the L0C bank that survives the
// call and sending the remaining 512 - `persistent_cols` columns out through
// the per-tile mailbox. The resident bank is flushed on the tile the caller
// marks last.
//
// `flags` is a bit set:
//   1  final tile of this program's queue -- drain the mailbox
//   2  first tile of a request -- (re)load A and initialize the accumulator
//   4  last tile of a request -- flush the resident columns to GM
//   8  prefetch the next tile's B into the following L1 bank; without it a
//      tile is loaded where it is consumed and `next_b` is not read
//
// `tile` selects the L1 bank, ringing over TLE_MMA_L1_B_BANKS.
//
// Shape contract: M is 64 and K is 512 (see mma_common.h). `bn` is the tile
// length and `persistent_cols` the resident width; both must be multiples of
// 16, and `persistent_cols` must not exceed what one L0C bank holds.
extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_mma_m64_resident_bf16(
    memref_t<__gm__ bfloat16_t, 2> *a, memref_t<__gm__ bfloat16_t, 2> *b,
    memref_t<__gm__ bfloat16_t, 2> *next_b, int64_t bn, int64_t kind,
    int64_t tile, int64_t flags, int64_t persistent_cols,
    memref_t<__gm__ float, 2> *out) {
  auto query = (__cbuf__ bfloat16_t *)get_imm(0);
  auto kv0 = query + 64 * 512;
  auto kv = kv0 + (tile % TLE_MMA_L1_B_BANKS) * bn * 512;
  auto weights = (__cbuf__ bfloat16_t *)get_imm(480 * 1024);
  auto persist = (__cc__ float *)get_imm(0);
  auto temporary = (__cc__ float *)get_imm(96 * 1024);
  auto ga = a->aligned + a->offset;
  auto go = out->aligned + out->offset;
  const bool final_tile = flags & 1;
  const bool first_request_tile = flags & 2;
  const bool last_request_tile = flags & 4;
  const bool prefetch = flags & 8;
  if (kind == 0) {
    if (tile == 0) {
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
    }
    if (first_request_tile) {
      if (tile != 0) {
        // The next request reuses the query bank. Explicitly release it on
        // MTE1 before MTE2 can overwrite the previous request's second panel.
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
      }
      gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_a(
          (__cbuf__ half *)query, (__gm__ half *)ga, 64, 64, 64, 512, 512,
          a->strides[0]);
    }
    if (prefetch) {
      if (tile == 0) {
        gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_b(
            (__cbuf__ half *)kv, (__gm__ half *)(b->aligned + b->offset), bn,
            bn, bn, 512, 512, b->strides[0]);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      }
      if (first_request_tile && tile != 0) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
      }
      // Consume the current bank's ready event before issuing the next DMA.
      // The QK MMA must not wait for that following tile's MTE2 operation.
      if (tile % 2 == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      else
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
      if (!final_tile) {
        auto next_kv = kv0 + ((tile + 1) % TLE_MMA_L1_B_BANKS) * bn * 512;
        gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> prefetch_b(
            (__cbuf__ half *)next_kv,
            (__gm__ half *)(next_b->aligned + next_b->offset), bn, bn, bn, 512,
            512, next_b->strides[0]);
        if (tile % 2 == 0)
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        else
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      }
      wait_mailbox(0);
      mma_m64<true, true>(query, kv, temporary, bn, 512, true);
    } else {
      // Load only this tile. Its producer completed before the weights for
      // tile-2 became ready; no readiness fence for a future tile is needed.
      gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_b(
          (__cbuf__ half *)kv, (__gm__ half *)(b->aligned + b->offset), bn, bn,
          bn, 512, 512, b->strides[0]);
      wait_mailbox(0);
      mma_m64<true>(query, kv, temporary, bn, 512, true);
    }
    store_mailbox(go, temporary, bn, out->strides[0], 0);
  } else {
    const uint32_t resident = (uint32_t)persistent_cols;
    const uint32_t spill = 512 - resident;
    gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_a(
        (__cbuf__ half *)weights, (__gm__ half *)ga, 64, 64, 64, bn, bn,
        a->strides[0]);
    mma_m64<false>(weights, kv, persist, resident, bn, first_request_tile);
    wait_mailbox(0);
    mma_m64<false>(weights, kv + resident * bn, temporary, spill, bn, true);
    store_mailbox(go + resident, temporary, spill, out->strides[0], 0);
    if (last_request_tile) {
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      copy_matrix_cc_to_gm(go, persist, 0, resident, 64, out->strides[0], 64, 0,
                           NoQuant, 0, false, true);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
    }
    if (final_tile)
      wait_mailbox(0);
  }
}
}
