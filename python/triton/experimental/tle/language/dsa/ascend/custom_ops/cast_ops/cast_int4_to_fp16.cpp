// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Based on CANN 9.1 dav_c220/kernel_operator_vec_vconv_impl.h:
// CastImpl and the int4-to-half CastIntrinsicsImpl specialization.
// Count mode, dst/src block strides 1/1, repeat strides 8/2.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_cast_int4_to_fp16(memref_t<__ubuf__ uint8_t, 1> *src,
                                      memref_t<__ubuf__ half, 1> *dst) {
  pipe_barrier(PIPE_ALL);
  set_mask_count();
  set_vector_mask(0, src->sizes[0] * 2);
  vconv_s42f16(dst->aligned + dst->offset, src->aligned + src->offset, 1, 1, 1,
               8, 2);
  set_mask_norm();
  set_vector_mask(~uint64_t(0), ~uint64_t(0));
  pipe_barrier(PIPE_ALL);
}
