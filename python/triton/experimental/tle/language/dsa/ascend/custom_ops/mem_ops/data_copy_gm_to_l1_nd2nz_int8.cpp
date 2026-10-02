// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Follows CANN 9.1 dav_c220/kernel_operator_data_copy_impl.h,
// DataCopyGM2L1ND2NZImplBase, int8_t branch. All dav_c220 Nd2NzParams
// fields are exposed. This primitive performs exactly one transfer; the
// caller owns bounds, padding, layout, and pipeline synchronization.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_data_copy_gm_to_l1_nd2nz_int8(
    memref_t<__gm__ int8_t, 2> *src, uint16_t nd_num, uint16_t n_value,
    uint16_t d_value, uint16_t src_nd_matrix_stride, uint16_t src_d_value,
    uint16_t dst_nz_c0_stride, uint16_t dst_nz_n_stride,
    uint16_t dst_nz_matrix_stride, memref_t<__cbuf__ int8_t, 4> *dst) {
  copy_gm_to_cbuf_multi_nd2nz_b8(
      dst->aligned + dst->offset, src->aligned + src->offset, 0, nd_num,
      n_value, d_value, src_nd_matrix_stride, src_d_value, dst_nz_c0_stride,
      dst_nz_n_stride, dst_nz_matrix_stride);
}
