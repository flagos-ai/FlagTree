// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// c310 replacements for the two mem-based Template helpers that the shared
// sort_base path still calls. The mem implementations live in
// lib/Vector/{Arange,Elementwise} and use vadds / PIPE_V barriers, which the
// A5 (c310) target does not provide. These scalar loops are only used by the
// BASE sort implementation; the A5-specific sort paths do not call them.

#include "Utils.h"
#include "Vector/VecUtils.h"
#include "Vector/Arange/ArangeUtils.h"

template <typename T>
__aiv__ void arange_1d(memref_t<__ubuf__ T, 1> *dst, int64_t offset,
                       int64_t stride) {
  auto *dst_ptr = dst->aligned + dst->offset;
  const int64_t n = dst->sizes[0];
  INTRINSIC(pipe_barrier, PIPE_ALL);
  for (int64_t i = 0; i < n; ++i) {
    dst_ptr[i] = static_cast<T>(offset + i * stride);
  }
  INTRINSIC(pipe_barrier, PIPE_ALL);
}

template <VectorOpTy OP, typename T>
__aiv__ void vector_eltwise_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                                  memref_t<__ubuf__ T, 1> *dst,
                                  memref_t<__ubuf__ T, 1> *tmp_buf,
                                  VectorLastAxisMode mode) {
  static_assert(OP == VectorOpTy::VADDS,
                "c310 compatibility helper only implements VADDS");
  (void)tmp_buf;
  (void)mode;
  auto *src_ptr = src0->aligned + src0->offset;
  auto *dst_ptr = dst->aligned + dst->offset;
  int64_t n = src0->sizes[0];
  if (dst->sizes[0] < n) {
    n = dst->sizes[0];
  }
  INTRINSIC(pipe_barrier, PIPE_ALL);
  for (int64_t i = 0; i < n; ++i) {
    dst_ptr[i] = static_cast<T>(src_ptr[i] + scalar);
  }
  INTRINSIC(pipe_barrier, PIPE_ALL);
}

template void arange_1d<int32_t>(memref_t<__ubuf__ int32_t, 1> *dst,
                                 int64_t offset, int64_t stride);

template void vector_eltwise_vs_1d<VectorOpTy::VADDS, float>(
    memref_t<__ubuf__ float, 1> *src0, float scalar,
    memref_t<__ubuf__ float, 1> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode);

template void vector_eltwise_vs_1d<VectorOpTy::VADDS, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, int32_t scalar,
    memref_t<__ubuf__ int32_t, 1> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode);

template void vector_eltwise_vs_1d<VectorOpTy::VADDS, int16_t>(
    memref_t<__ubuf__ int16_t, 1> *src0, int16_t scalar,
    memref_t<__ubuf__ int16_t, 1> *dst, memref_t<__ubuf__ int16_t, 1> *tmp_buf,
    VectorLastAxisMode mode);
