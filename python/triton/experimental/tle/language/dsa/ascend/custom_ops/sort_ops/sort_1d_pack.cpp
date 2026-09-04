// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIV/Vector) into its own bitcode.
#include "sort_base.h"
#include "sort_s4096_k129_512.h"
#include "sort_s4096_k1_128_k2048.h"

extern "C" {

__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_sort_1d_pack_float(
    memref_t<__ubuf__ float, 1> *src, memref_t<__ubuf__ float, 1> *tmp_buf,
    bool descending, int64_t topk, int64_t index_offset, int64_t sort_impl,
    memref_t<__ubuf__ float, 1> *dst_proposals) {
  switch (sort_impl) {
  case 0:
    sort_base_impl(src, tmp_buf, descending, topk, index_offset, dst_proposals);
    return;
  case 1:
    sort_s4096_k129_512_impl(src, tmp_buf, descending, topk, index_offset,
                             dst_proposals);
    return;
  case 2:
    sort_s4096_k1_128_k2048_impl(src, tmp_buf, descending, topk, index_offset,
                                 dst_proposals);
    return;
  default:
    sort_base_impl(src, tmp_buf, descending, topk, index_offset, dst_proposals);
    return;
  }
}

} // extern "C"
