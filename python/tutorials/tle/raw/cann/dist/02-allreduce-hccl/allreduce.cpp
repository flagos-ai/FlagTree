// Copyright 2026- FlagOS Contributors
//
// Minimal AscendC device HCCL AllReduce (sum, in-place).
// Only AIV block 0 drives HCCL; mirrors MatmulAllReduceBase notifyFlag_.

#include "Utils.h"
#include "kernel_operator.h"
#include "lib/hccl/hccl.h"

using namespace AscendC;

extern "C" {

// inputs: buf, n, ctx_u64, status | output: dummy
__aiv__ void _mlir_ciface_hccl_allreduce(memref_t<__gm__ half, 1> *buf,
                                         int64_t n,
                                         memref_t<__gm__ uint64_t, 1> *ctx_u64,
                                         memref_t<__gm__ int32_t, 1> *status,
                                         memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  if (GetBlockIdx() != 0) {
    return;
  }

  __gm__ half *data = buf->aligned + buf->offset;
  __gm__ uint64_t *ctx_box = ctx_u64->aligned + ctx_u64->offset;
  __gm__ int32_t *st = status->aligned + status->offset;

  if (n <= 0) {
    *st = -1;
    return;
  }

  GM_ADDR context = reinterpret_cast<GM_ADDR>(*ctx_box);
  if (context == nullptr) {
    *st = -2;
    return;
  }
  SetHcclContext<0>(reinterpret_cast<__gm__ uint8_t *>(context));
  if (GetHcclContext<0>() != context) {
    *st = -3;
    return;
  }

  Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
  hccl.Init(context);

  HcclHandle handle = hccl.AllReduce(
      reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(data),
      static_cast<uint64_t>(n), AscendC::HCCL_DATA_TYPE_FP16,
      AscendC::HCCL_REDUCE_SUM, /*repeat=*/1);
  if (handle < 0) {
    *st = -4;
    hccl.Finalize();
    return;
  }

  hccl.Commit(handle);
  hccl.Wait(handle);
  hccl.Finalize();
  *st = 0;
}

} // extern "C"
