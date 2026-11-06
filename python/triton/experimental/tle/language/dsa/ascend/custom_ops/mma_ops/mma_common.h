// Copyright 2026- Xcoresigma Technology Co., Ltd

// Shared by the two M64 cube operators: one MMA over a single L0C bank, with
// a per-tile GM mailbox for the columns the bank cannot hold, and the flags
// that hand that mailbox back and forth with the vector core.
//
// The K pipeline follows Huawei's Apache-2.0 DefaultMmaTile (2024), with an
// explicit L0C pointer, accumulator initialization and an actual N extent.
//
// Shape contract. M is 64 and K is 512: both are woven into the capacity
// arithmetic here (the 16 KiB L0 operand halves, `half_l0 / rows / 16 * 16`)
// and into the callers' L1 offsets, so they are fixed rather than passed.
// N is the caller's `n`.
#ifndef TLE_ASCEND_CUSTOM_OPS_MMA_COMMON_H
#define TLE_ASCEND_CUSTOM_OPS_MMA_COMMON_H

#include "Macro/common.h"
#include "Macro/coord.h"
#include "Macro/operators/iterator.h"
#include "Macro/operators/iterators/gm_to_l1_iterator.h"
#include "Macro/operators/iterators/l1_to_l0_iterator.h"
#include "Macro/operators/mma.h"
#include "Macro/params.h"
#include "Utils.h"

// The resident operator rings its B tiles over this many L1 banks.
#define TLE_MMA_L1_B_BANKS 3

namespace {
constexpr ArchType arch = ArchType::ASCEND_V220;

template <bool transpose_b, bool inputs_ready = false>
__aicore__ __attribute__((always_inline)) void
mma_m64(__cbuf__ bfloat16_t *a, __cbuf__ bfloat16_t *b, __cc__ float *c,
        uint32_t n, uint32_t k, bool initialize) {
  using LoadA =
      l1_to_l0_a<arch, bfloat16_t, false, CubeFormat::ZN, CubeFormat::ZZ>;
  using LoadB =
      l1_to_l0_b<arch, bfloat16_t, transpose_b, CubeFormat::ZN, CubeFormat::NZ>;
  using Mma = mmad<arch, bfloat16_t, bfloat16_t, float, false>;
  constexpr uint32_t rows = 64, half_l0 = 16384;
  auto ca = (__ca__ bfloat16_t *)get_imm(0);
  auto cb = (__cb__ bfloat16_t *)get_imm(0);
  uint32_t part_k = min(half_l0 / rows / 16 * 16, half_l0 / n / 16 * 16);
  uint32_t parts = (k + part_k - 1) / part_k;
  if constexpr (!inputs_ready) {
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  }
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (uint32_t i = 0; i < parts; ++i) {
    uint32_t actual_k = min(part_k, k - i * part_k);
    auto event = i % 2 ? EVENT_ID1 : EVENT_ID0;
    auto pa = ca + i % 2 * half_l0;
    auto pb = cb + i % 2 * half_l0;
    if constexpr (!inputs_ready)
      if (i == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, event);
    LoadA(pa, a + i * part_k * rows, rows, actual_k, 1, rows / 16,
          actual_k / 16, 1);
    if constexpr (!inputs_ready) {
      if (i + 1 == parts)
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      if (i == 0)
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    }
    if constexpr (transpose_b) {
      LoadB(pb, b + i * part_k * n, n, actual_k, 1, n / 16, 1, n / 16);
    } else {
      LoadB(pb, b + i * part_k * 16, n, actual_k, k / 16, 1, 1, n / 16);
    }
    if constexpr (!inputs_ready)
      if (i + 1 == parts)
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, event);
    wait_flag(PIPE_MTE1, PIPE_M, event);
    Mma(c, pa, pb, rows, n, actual_k, initialize && i == 0);
    // Native arch22 only drains M for fewer than ten 16x16 output blocks.
    // M/MTE1 ping-pong events already protect the two L0 operand banks.
    if ((rows / 16) * (n / 16) < 10)
      pipe_barrier(PIPE_M);
    set_flag(PIPE_M, PIPE_MTE1, event);
  }
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  if constexpr (!inputs_ready) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  }
}

__aicore__ __attribute__((always_inline)) void wait_mailbox(uint32_t bank) {
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
}

__aicore__ __attribute__((always_inline)) void
store_mailbox(__gm__ float *dst, __cc__ float *src, uint32_t n, uint32_t stride,
              uint32_t bank) {
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  copy_matrix_cc_to_gm(dst, src, 0, n, 64, stride, 64, 0, NoQuant, 0, false,
                       true);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
}
} // namespace

#endif // TLE_ASCEND_CUSTOM_OPS_MMA_COMMON_H
