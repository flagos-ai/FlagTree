// Copyright 2026- FlagOS Contributors
// WINDOW_IN AllReduce aligned to matmul_all_reduce_base.h:
//   phase 0: Init + AllReduce(Prepare)  → handle in hbox, Hccl bytes in skel_buf
//   phase 1: Commit(turn)               → PostProcEachTurn (MM already wrote WINDOW)
//   phase 2: Wait + Finalize
//
// Host passes windowsIn as a Triton ptr; MM tl.store goes straight into WINDOW
// (same as ops: addrs_->cGM = GetWindowsInAddr). No out→window GmCopy.

#include "Utils.h"
#include "kernel_operator.h"
#include "lib/hccl/hccl.h"

#include <stdint.h>

using namespace AscendC;

constexpr int64_t kBufferWindowIn = 2;
using HcclAicpu = Hccl<HCCL_SERVER_TYPE_AICPU>;

__aicore__ inline void StoreHcclToGm(__gm__ uint8_t *dst, const HcclAicpu &hccl) {
  const uint8_t *src = reinterpret_cast<const uint8_t *>(&hccl);
  for (uint32_t i = 0; i < sizeof(HcclAicpu); ++i) {
    dst[i] = src[i];
  }
}

__aicore__ inline void LoadHcclFromGm(HcclAicpu &hccl, __gm__ uint8_t *src) {
  uint8_t *dst = reinterpret_cast<uint8_t *>(&hccl);
  for (uint32_t i = 0; i < sizeof(HcclAicpu); ++i) {
    dst[i] = src[i];
  }
}

extern "C" {

// args: out, n_elems, tile_cnt, use_buffer_type, phase, turn_idx,
//       ctx, hbox, status, skel_buf | dummy
// phase: 0=Prepare, 1=Commit(turn_idx), 2=Wait+Finalize
// out = AllReduce dst (user buffer); WINDOW is filled by Triton MM.
__aiv__ void _mlir_ciface_hccl_reduce_skel(
    memref_t<__gm__ half, 1> *out, int64_t n_elems, int64_t tile_cnt,
    int64_t use_buffer_type, int64_t phase, int64_t turn_idx,
    memref_t<__gm__ uint64_t, 1> *ctx_u64,
    memref_t<__gm__ int64_t, 1> *hbox,
    memref_t<__gm__ int32_t, 1> *status,
    memref_t<__gm__ int64_t, 1> *skel_buf,
    memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  if (GetBlockIdx() != 0) {
    return;
  }

  __gm__ int32_t *st = status->aligned + status->offset;
  __gm__ int64_t *hb = hbox->aligned + hbox->offset;
  __gm__ uint8_t *skel = reinterpret_cast<__gm__ uint8_t *>(skel_buf->aligned + skel_buf->offset);

  if (n_elems <= 0 || tile_cnt <= 0 || (n_elems % tile_cnt) != 0) {
    *st = -1;
    return;
  }
  if (use_buffer_type != kBufferWindowIn) {
    *st = -3;
    return;
  }

  GM_ADDR context = reinterpret_cast<GM_ADDR>(*(ctx_u64->aligned + ctx_u64->offset));
  if (context == nullptr) {
    *st = -2;
    return;
  }

  GM_ADDR outputGM = reinterpret_cast<GM_ADDR>(out->aligned + out->offset);
  const uint64_t per = static_cast<uint64_t>(n_elems / tile_cnt);

  // Working image in a byte buffer — never ~HcclAicpu a live client before Wait.
  alignas(HcclAicpu) uint8_t local_bytes[sizeof(HcclAicpu)];
  HcclAicpu &hccl = *reinterpret_cast<HcclAicpu *>(local_bytes);

  if (phase == 0) {
    {
      HcclAicpu seed;
      const uint8_t *src = reinterpret_cast<const uint8_t *>(&seed);
      for (uint32_t i = 0; i < sizeof(HcclAicpu); ++i) {
        local_bytes[i] = src[i];
      }
    }
    SetHcclContext<0>(reinterpret_cast<__gm__ uint8_t *>(context));
    hccl.Init(GetHcclContext<0>());
    // ops: addrs_->cGM = GetWindowsInAddr; AllReduce(cGM, outputGM, per, ..., tileCnt)
    GM_ADDR cGM = hccl.GetWindowsInAddr(hccl.GetRankId());
    if (cGM == nullptr) {
      *st = -7;
      hccl.Finalize();
      return;
    }
    HcclHandle h = hccl.AllReduce(cGM, outputGM, per, AscendC::HCCL_DATA_TYPE_FP16,
                                   AscendC::HCCL_REDUCE_SUM, static_cast<uint8_t>(tile_cnt));
    if (h < 0) {
      *st = -4;
      hccl.Finalize();
      return;
    }
    StoreHcclToGm(skel, hccl);
    *hb = static_cast<int64_t>(h);
    *st = 0;
    return;
  }

  HcclHandle h = static_cast<HcclHandle>(*hb);
  if (h < 0) {
    *st = -5;
    return;
  }

  LoadHcclFromGm(hccl, skel);

  if (phase == 1) {
    // ops PostProcEachTurn: Mc2SyncAll (done in Triton) + Commit — no copy.
    if (turn_idx < 0 || turn_idx >= tile_cnt) {
      *st = -6;
      return;
    }
    (void)per;
    PipeBarrier<PIPE_ALL>();
    hccl.Commit(h);
    StoreHcclToGm(skel, hccl);
    *st = 0;
    return;
  }

  if (phase == 2) {
    hccl.Wait(h);
    PipeBarrier<PIPE_ALL>();
    hccl.Finalize();
    *hb = -1;
    *st = 0;
    return;
  }

  *st = -8;
}

} // extern "C"
