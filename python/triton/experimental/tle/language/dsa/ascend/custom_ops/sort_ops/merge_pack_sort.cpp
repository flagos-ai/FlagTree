// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIV/Vector) into its own bitcode.
/**
 * =====================================================================
 *
 *   - This file uses vmrgsort4 in *truncated/exhaustion mode*
 *     (config[12] = isAllStored = 1): the merge stops as soon as any way's
 *     on-chip data is exhausted, without requiring all 4 ways to drain.
 *   - After each vmrgsort4, VMS4_SR is read back via get_vms4_sr(). The
 *     register packs the actually consumed proposal count of each way into
 *     16-bit fields:
 *         VMS4_SR[15:0]  = consumed by way 0
 *         VMS4_SR[31:16] = consumed by way 1
 *         VMS4_SR[47:32] = consumed by way 2
 *         VMS4_SR[63:48] = consumed by way 3
 *   - Software advances each way's read cursor and remaining count from
 *     these values, and writes the consumed counts out to decide "how much
 *     to load from each way next time, and from where".
 *
 * Data format:
 *   proposal = [value (f32), index (i32 reinterpreted as f32)], 8 bytes.
 *   Input src_proposals[N*2] f32, segmented into blocks of block_size
 *   proposals, descending within each block.
 *
 * config bit fields (confirmed by the vmrgsort4 wrapper in intrisics.h):
 *   config = (repeat & 0xff) | ((maskSignal & 0xf) << 8) | ((isAllStored & 1)
 * << 12) maskSignal: 0x3=2 ways, 0x7=3 ways, 0xf=4 ways
 *
 */

#include "DMA/DMAUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"
#include "sort_common.h"

extern "C" {

///
/// Fully controlled from the Python side: which ways, each way's starting
/// proposal offset in src, each way's length, and the output buffer. The C++
/// side only performs a single (exhaustion-mode) vmrgsort4 and reports the
/// per-way consumed counts. The multi-level merge loop, buffer allocation
/// and cursor advancement are all composed from this primitive on the
/// Python side.
///
/// Note: the function returns void; outputs are written back through the
/// dst_proposals / consumed_out pointers.
///
/// src_proposals: [in] UB proposal buffer (all ways live in it; starts given by
/// off*) ways:          [in] number of active ways (2/3/4) off0..off3:    [in]
/// starting proposal offset of each way (in proposals; unused ways pass 0)
/// len0..len3:    [in] length of each way in proposals (unused ways pass 0)
/// dst_proposals: [out, written via pointer] merged output (safe prefix first)
/// consumed_out:  [out, written via pointer] tensor<4 x i32> — per-way consumed
/// counts this round (decoded from VMS4_SR)
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge_exhaust_sort4_float(
    memref_t<__ubuf__ float, 1> *src_proposals, int64_t ways, int64_t off0,
    int64_t off1, int64_t off2, int64_t off3, int64_t len0, int64_t len1,
    int64_t len2, int64_t len3, memref_t<__ubuf__ float, 1> *dst_proposals,
    memref_t<__ubuf__ int32_t, 1> *consumed_out) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto dst_ptr = dst_proposals->aligned + dst_proposals->offset;
  auto cons_ptr = consumed_out->aligned + consumed_out->offset;

  int64_t offs[4] = {off0, off1, off2, off3};
  int64_t lns[4] = {len0, len1, len2, len3};

  // ★ Key point: compact the non-empty ways into consecutive lanes
  //   0..(active-1). vmrgsort4_exhaust uses mask = (1<<ways)-1 and assumes
  //   the active ways are laid out contiguously from lane 0. But the caller
  //   places the 4 ways in fixed slots and marks empty ways with length 0;
  //   merging tail segments can leave holes ("a middle way exhausted
  //   first", e.g. l0>0, l1=0, l2>0). Feeding the fixed slots directly
  //   would make the mask select a zero-length lane -> vmrgsort4 illegal
  //   configuration (VEC illegal config). So collect only the ways with
  //   len>0 into consecutive lanes, record their original indices, and
  //   scatter each lane's consumed count back to its original slot after
  //   the merge.
  __ubuf__ float *xn[4];
  uint32_t lens[4] = {0, 0, 0, 0};
  int orig_idx[4] = {0, 0, 0, 0}; // compacted lane w -> original slot
  int active = 0;
  for (int i = 0; i < 4; ++i) {
    if (lns[i] > 0) {
      xn[active] = src_ptr + offs[i] * npp;
      lens[active] = (uint32_t)lns[i];
      orig_idx[active] = i;
      active++;
    }
  }

  // Zero the consumed counts of all 4 ways first (empty ways always
  // consume 0)
  for (int i = 0; i < 4; ++i)
    *(cons_ptr + i) = 0;

  if (active == 0) {
    return; // no data, no-op
  }

  if (active == 1) {
    // A single way left: no merge needed, copy the remainder straight to
    // dst (standard k-way merge tail optimization). Note: vmrgsort4 does
    // not support 1 way (mask=0x1 is illegal), so it must be special-cased.
    int64_t cnt = lens[0];
    memref_t<__ubuf__ float, 1> from{src_proposals->aligned,
                                     src_proposals->allocated,
                                     src_proposals->offset +
                                         offs[orig_idx[0]] * npp,
                                     {cnt * npp},
                                     {1}};
    memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                   dst_proposals->allocated,
                                   dst_proposals->offset,
                                   {cnt * npp},
                                   {1}};
    copy_ubuf_to_ubuf_1d_core(&from, &to);
    *(cons_ptr + orig_idx[0]) = (int32_t)cnt;
    return;
  }

  // active >= 2: normal exhaustion merge (lanes contiguous after
  // compaction, mask is valid)
  uint64_t sr = vmrgsort4_exhaust<float>(dst_ptr, xn, lens, active);

  // Scatter the consumed count of compacted lane w back to its original
  // slot orig_idx[w]
  for (int w = 0; w < active; ++w)
    *(cons_ptr + orig_idx[w]) = (int32_t)vms4_consumed(sr, w);
}

} // extern "C"
