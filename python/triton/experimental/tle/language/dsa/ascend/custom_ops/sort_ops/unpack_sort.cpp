// Copyright 2026- Xcoresigma Technology Co., Ltd

// Compiled standalone (AIV/Vector) into its own bitcode.
#include "sort_common.h"

extern "C" {

/// unpack: the first topk proposals -> value/index (for the final output)
///
/// Note: the function returns void; outputs are written back through the
/// dst_value / dst_index pointers.
/// src_proposals: [in]  the first topk sorted proposals
/// topk:          [in]  K
/// dst_value:     [out, written via pointer] tensor<topk x f32>
/// dst_index:     [out, written via pointer] tensor<topk x i32>
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_unpack_sort_float(
    memref_t<__ubuf__ float, 1> *src_proposals, int64_t topk,
    memref_t<__ubuf__ float, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  // CRITICAL: vreducev2_1d_with_pattern_mode uses src->sizes[0] to set mask,
  // NOT the topk parameter. If src buffer is larger than topk*npp (which it
  // always is — it's the full UNPACK_CHUNK*2 buffer), vreducev2 will process
  // garbage past the valid data and may trigger "illegal configurations".
  // Solution: create a src view limited to exactly topk*npp elements.
  memref_t<__ubuf__ float, 1> src_view{src_proposals->aligned,
                                       src_proposals->allocated,
                                       src_proposals->offset,
                                       {topk * npp},
                                       {1}};
  memref_t<__ubuf__ float, 1> dval{
      dst_value->aligned, dst_value->allocated, dst_value->offset, {topk}, {1}};
  memref_t<__ubuf__ int32_t, 1> didx{
      dst_index->aligned, dst_index->allocated, dst_index->offset, {topk}, {1}};
  unpack_proposals(&src_view, &dval, &didx, topk);
}

} // extern "C"
