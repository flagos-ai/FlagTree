// Copyright 2026- FlagOS Contributors
//
// Device AllReduce (AscendC Hccl AICPU) — MatmulAllReduceBase reduce-only:
//   Init → (WINDOW_IN: cGM=GetWindowsInAddr) → AllReduce(prepare)
//   → copy user→window (stand-in for MM writes)
//   → for each tile: Commit
//   → Wait → Finalize
//
// Host owns KFC start/finish (HcclKfcSession). This entry is one full
// device session; only block 0 drives HCCL (notify-only).

#include "Utils.h"
#include "kernel_operator.h"
#include "lib/hccl/hccl.h"

#include <stdint.h>

using namespace AscendC;

constexpr int64_t kBufferWindowIn = 2;
constexpr uint32_t kCopyUbBytes = 32U * 1024U;

__aicore__ inline void GmCopyHalf(GM_ADDR dst, GM_ADDR src, uint64_t nElems) {
  if (nElems == 0) {
    return;
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> tbuf;
  pipe.InitBuffer(tbuf, kCopyUbBytes);
  LocalTensor<half> lt = tbuf.Get<half>();
  const uint64_t maxElems = static_cast<uint64_t>(kCopyUbBytes / sizeof(half));
  GlobalTensor<half> gsrc;
  GlobalTensor<half> gdst;
  for (uint64_t off = 0; off < nElems; off += maxElems) {
    uint64_t cur = nElems - off;
    if (cur > maxElems) {
      cur = maxElems;
    }
    uint64_t aligned = (cur + 15) & ~static_cast<uint64_t>(15);
    if (aligned > maxElems) {
      aligned = maxElems;
      cur = aligned;
      if (off + cur > nElems) {
        cur = nElems - off;
        aligned = (cur + 15) & ~static_cast<uint64_t>(15);
        if (aligned > maxElems) {
          aligned = maxElems;
        }
      }
    }
    gsrc.SetGlobalBuffer((__gm__ half *)(src) + off, aligned);
    gdst.SetGlobalBuffer((__gm__ half *)(dst) + off, aligned);
    DataCopy(lt, gsrc, static_cast<uint32_t>(aligned));
    event_t ev =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(ev);
    WaitFlag<HardEvent::MTE2_MTE3>(ev);
    DataCopy(gdst, lt, static_cast<uint32_t>(aligned));
    event_t ev2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(ev2);
    WaitFlag<HardEvent::MTE3_MTE2>(ev2);
  }
}

extern "C" {

// args: buf, n_elems, tile_cnt, use_buffer_type, do_copy, ctx, status | dummy
__aiv__ void _mlir_ciface_hccl_reduce_ar(memref_t<__gm__ half, 1> *buf,
                                         int64_t n_elems, int64_t tile_cnt,
                                         int64_t use_buffer_type,
                                         int64_t do_copy,
                                         memref_t<__gm__ uint64_t, 1> *ctx_u64,
                                         memref_t<__gm__ int32_t, 1> *status,
                                         memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  if (GetBlockIdx() != 0) {
    return;
  }

  __gm__ half *out_ptr = buf->aligned + buf->offset;
  __gm__ uint64_t *ctx_box = ctx_u64->aligned + ctx_u64->offset;
  __gm__ int32_t *st = status->aligned + status->offset;

  if (n_elems <= 0 || tile_cnt <= 0 || (n_elems % tile_cnt) != 0) {
    *st = -1;
    return;
  }
  GM_ADDR context = reinterpret_cast<GM_ADDR>(*ctx_box);
  if (context == nullptr) {
    *st = -2;
    return;
  }

  SetHcclContext<0>(reinterpret_cast<__gm__ uint8_t *>(context));
  Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
  hccl.Init(GetHcclContext<0>());

  const uint64_t per = static_cast<uint64_t>(n_elems / tile_cnt);
  const uint64_t cOffset = per * sizeof(half);

  GM_ADDR outputGM = reinterpret_cast<GM_ADDR>(out_ptr);
  GM_ADDR cGM = outputGM;
  if (use_buffer_type == kBufferWindowIn) {
    cGM = hccl.GetWindowsInAddr(hccl.GetRankId());
    if (cGM == nullptr) {
      *st = -7;
      hccl.Finalize();
      return;
    }
  }

  HcclHandle handle =
      hccl.AllReduce(cGM, outputGM, per, AscendC::HCCL_DATA_TYPE_FP16,
                     AscendC::HCCL_REDUCE_SUM, static_cast<uint8_t>(tile_cnt));
  if (handle < 0) {
    *st = -4;
    hccl.Finalize();
    return;
  }

  if (use_buffer_type == kBufferWindowIn && do_copy != 0) {
    GmCopyHalf(cGM, outputGM, static_cast<uint64_t>(n_elems));
  }

  GM_ADDR cTurn = cGM;
  for (int64_t t = 0; t < tile_cnt; ++t) {
    PipeBarrier<PIPE_ALL>();
    hccl.Commit(handle);
    cTurn =
        reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(cTurn) + cOffset);
    (void)cTurn;
  }

  hccl.Wait(handle);
  PipeBarrier<PIPE_ALL>();
  hccl.Finalize();
  *st = 0;
}

} // extern "C"
