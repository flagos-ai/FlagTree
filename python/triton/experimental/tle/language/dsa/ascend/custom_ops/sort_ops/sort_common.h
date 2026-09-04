// Copyright 2026- Xcoresigma Technology Co., Ltd

// Shared vmrgsort4 / proposal helpers, included by the other sort_ops files.

#pragma once

#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"

// Exhaustion mode: config[12] = isAllStored = 1
constexpr uint64_t VMRG4_EXHAUST_BIT = (uint64_t)1 << 12;

// Read VMS4_SR (per-way consumed counts, packed in 16-bit fields).
// See get_vms4_sr at intrisics.h #2028.
static inline __aiv__ __attribute__((always_inline)) uint64_t read_vms4_sr() {
  return (uint64_t)get_vms4_sr();
}

// Extract the consumed proposal count of way i (0..3) from the packed register
static inline __aiv__ __attribute__((always_inline)) uint32_t
vms4_consumed(uint64_t sr, int i) {
  return (uint32_t)((sr >> (16 * i)) & 0xFFFF);
}

// ============================================================================
// One exhaustion-mode K-way merge (K=2/3/4); returns VMS4_SR.
//   dst:  start of the output proposals
//   xn:   array of start pointers of the K input ways
//   lens: input length of each way (in proposals)
//   ways: 2/3/4
// Output length = sum of per-way consumed counts (= the safe prefix emitted
// until some way is exhausted).
// ============================================================================
template <typename T>
__aiv__ __attribute__((always_inline)) uint64_t vmrgsort4_exhaust(
    __ubuf__ T *dst, __ubuf__ T **xn, const uint32_t *lens, int ways) {
  // src1 = per-way lengths packed into 4x16-bit fields (unused ways filled
  // with 0)
  uint64_t xm = 0;
  for (int i = 0; i < ways; ++i)
    xm |= ((uint64_t)(lens[i] & 0xFFFF)) << (16 * i);

  // maskSignal: `ways` active ways -> low `ways` bits set
  uint64_t mask = ((1ull << ways) - 1) & 0xF;
  uint64_t config =
      (mask << 8) | VMRG4_EXHAUST_BIT | 1; // repeat=1, exhaustion mode

  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmrgsort4, dst, xn, xm, config);

  // Key point: read back the per-way consumed counts (feedback channel
  // exclusive to exhaustion mode)
  return read_vms4_sr();
}

// Split compact proposals ([value (f32), index (i32 reinterpreted as f32)]
// pairs) into separate value/index sequences. Extract the index first so it
// is not clobbered by the value write.
static inline __aiv__ __attribute__((always_inline)) void
unpack_proposals(memref_t<__ubuf__ float, 1> *src,
                 memref_t<__ubuf__ float, 1> *dst_value,
                 memref_t<__ubuf__ int32_t, 1> *dst_index, int64_t real_num) {
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, real_num);
  memref_t<__ubuf__ int32_t, 1> src_int32;
  view_as<float, int32_t, 1>(src, &src_int32);
  vreducev2_1d_with_pattern_mode<int32_t, PatternMode::INDEX_1_FROM_2_ELEMENTS>(
      &src_int32, dst_index);
  vreducev2_1d_with_pattern_mode<float, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
      src, dst_value);
  INTRINSIC_NO_ARGS(set_mask_norm);
}
