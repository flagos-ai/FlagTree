// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Based on CANN 9.1 dav_c220/kernel_operator_vec_vconv_impl.h: CastImpl
// (Cast Level 2) and the int8_t -> half CastIntrinsicsImpl specialization,
// which only supports RoundMode::CAST_NONE on this device. Count mode with a
// single repeat; dst/src block strides 1/1, repeat strides 8/4
// ({1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2}).
// Pipeline barriers are the caller's responsibility; this op adds none.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_cast_int8_to_fp16(memref_t<__ubuf__ int8_t, 1> *src,
                                      int32_t round_mode, uint32_t count,
                                      memref_t<__ubuf__ half, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert((round_mode == 0) &&
         "cast_int8_to_fp16: only CAST_NONE is supported from int8_t to half "
         "on this device");
#endif
  (void)round_mode;
  set_mask_count();
  set_vector_mask(0, count);
  vconv_s82f16(dst->aligned + dst->offset, src->aligned + src->offset, 1, 1, 1,
               8, 4);
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
