// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIC/Cube) into its own bitcode.
#include "mma_common.h"

// One M64 MMA per call, with both operands double-buffered in L1.
//
// `kind` 0 multiplies a [64, 512] A tile by a [n, 512] B tile transposed,
// which is a query against a key tile; `kind` 1 multiplies a [64, n] A tile
// by the B tile already in L1, which is the weights against the same values.
// Pairing the two kinds on one B residency is the point: kind 1 does not
// reload B, so a tile crosses GM once.
//
// `step` selects the L1 bank pair, so the caller can issue the next tile's
// loads while this one is still in the cube. `final_step` drains the mailbox
// on the last call.
//
// Shape contract: M is 64 and K is 512 (see mma_common.h). `n` is the tile
// length and must be a multiple of 16.
extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_mma_m64_tile_bf16(memref_t<__gm__ bfloat16_t, 2> *a,
                                      memref_t<__gm__ bfloat16_t, 2> *b,
                                      int64_t n, int64_t kind, int64_t step,
                                      int64_t final_step,
                                      memref_t<__gm__ float, 2> *out) {
  const uint32_t bn = (uint32_t)n;
  auto query0 = (__cbuf__ bfloat16_t *)get_imm(0);
  auto query = query0 + (step % 2) * 64 * 512;
  auto kv = query0 + 2 * 64 * 512 + (step % 2) * bn * 512;
  auto weights = (__cbuf__ bfloat16_t *)get_imm(480 * 1024);
  auto temporary = (__cc__ float *)get_imm(0);
  auto ga = a->aligned + a->offset;
  auto go = out->aligned + out->offset;
  if (kind == 0) {
    if (step == 0)
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
    gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_a(
        (__cbuf__ half *)query, (__gm__ half *)ga, 64, 64, 64, 512, 512,
        a->strides[0]);
    gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_b(
        (__cbuf__ half *)kv, (__gm__ half *)(b->aligned + b->offset), bn, bn,
        bn, 512, 512, b->strides[0]);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_mailbox(0);
    mma_m64<true, true>(query, kv, temporary, bn, 512, true);
    store_mailbox(go, temporary, bn, out->strides[0], 0);
  } else {
    gm_to_l1<arch, half, CubeFormat::ND, CubeFormat::ZN> copy_a(
        (__cbuf__ half *)weights, (__gm__ half *)ga, 64, 64, 64, bn, bn,
        a->strides[0]);
    wait_mailbox(0);
    mma_m64<false>(weights, kv, temporary, 512, bn, true);
    store_mailbox(go, temporary, 512, out->strides[0], 0);
    if (final_step)
      wait_mailbox(0);
  }
}
}
