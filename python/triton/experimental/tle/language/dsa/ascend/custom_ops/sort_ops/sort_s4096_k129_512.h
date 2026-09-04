// Copyright 2026- Xcoresigma Technology Co., Ltd

// Included only by custom_ops_aiv.cpp.
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"
#include "sort_common.h"

constexpr int64_t CHUNK = 1024;
constexpr int64_t WAYS = 4;
constexpr int64_t SMALLK_NPP = PROPOSALS_BYTES / sizeof(float);

__aiv__ __attribute__((always_inline)) void
prepare_index_smallk(memref_t<__ubuf__ int32_t, 1> *src_index, int64_t real_num,
                     int64_t sort_num, int64_t index_offset) {
  src_index->sizes[0] = sort_num;
  arange_1d(src_index, index_offset, 1);
  src_index->sizes[0] = real_num;
}

__aiv__ __attribute__((always_inline)) void
prepare_desc_value_smallk(memref_t<__ubuf__ float, 1> *src, int64_t real_num,
                          int64_t sort_num) {
  auto src_ptr = src->aligned + src->offset;
  int64_t fill_num = sort_num - real_num;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < fill_num; ++i) {
    *(src_ptr + real_num + i) = static_cast<float>(FLOAT_NEG_INF);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) void
block_sort_smallk(memref_t<__ubuf__ float, 1> *src_value,
                  memref_t<__ubuf__ int32_t, 1> *src_index,
                  memref_t<__ubuf__ float, 1> *dst, int64_t sort_num) {
  int64_t repeat = sort_num / BIT_SORT_NUM_PER_REPEAT;
  int64_t sort_num_per_intrinsic =
      INTR_MAX_REPEAT_CNTS * BIT_SORT_NUM_PER_REPEAT;
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_value_ptr = src_value->aligned + src_value->offset;
  auto src_index_ptr = src_index->aligned + src_index->offset;

  if (repeat >= INTR_MAX_REPEAT_CNTS) {
    for (int64_t i = 0; i < repeat / INTR_MAX_REPEAT_CNTS; ++i) {
      INTRINSIC(
          vbitsort, dst_ptr + i * sort_num_per_intrinsic * SMALLK_NPP,
          src_value_ptr + i * sort_num_per_intrinsic,
          (__ubuf__ uint32_t *)(src_index_ptr + i * sort_num_per_intrinsic),
          INTR_MAX_REPEAT_CNTS);
    }
  }

  if (repeat % INTR_MAX_REPEAT_CNTS != 0) {
    int64_t loop_num = repeat / INTR_MAX_REPEAT_CNTS;
    INTRINSIC(vbitsort,
              dst_ptr + loop_num * sort_num_per_intrinsic * SMALLK_NPP,
              src_value_ptr + loop_num * sort_num_per_intrinsic,
              (__ubuf__ uint32_t *)(src_index_ptr +
                                    loop_num * sort_num_per_intrinsic),
              repeat % INTR_MAX_REPEAT_CNTS);
  }
}

__aiv__ __attribute__((always_inline)) void
merge_sort_1024_smallk(memref_t<__ubuf__ float, 1> *src,
                       memref_t<__ubuf__ float, 1> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  // Round 1: thirty-two 32-proposal VBS runs -> eight 128-proposal runs.
  {
    __ubuf__ float *xn[4] = {
        src_ptr,
        src_ptr + 32 * SMALLK_NPP,
        src_ptr + 64 * SMALLK_NPP,
        src_ptr + 96 * SMALLK_NPP,
    };
    uint64_t xm = 32 | (32ull << 16) | (32ull << 32) | (32ull << 48);
    uint64_t config = WAY4_CONFIG_MODE | 8;
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(vmrgsort4, dst_ptr, xn, xm, config);
  }

  // Round 2: eight 128-proposal runs -> two 512-proposal runs.
  {
    __ubuf__ float *xn[4] = {
        dst_ptr,
        dst_ptr + 128 * SMALLK_NPP,
        dst_ptr + 256 * SMALLK_NPP,
        dst_ptr + 384 * SMALLK_NPP,
    };
    uint64_t xm = 128 | (128ull << 16) | (128ull << 32) | (128ull << 48);
    uint64_t config = WAY4_CONFIG_MODE | 2;
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(vmrgsort4, src_ptr, xn, xm, config);
  }

  // Round 3: two 512-proposal runs -> one 1024-proposal run.
  {
    __ubuf__ float *xn[2] = {
        src_ptr,
        src_ptr + 512 * SMALLK_NPP,
    };
    uint64_t xm = 512 | (512ull << 16);
    uint64_t config = WAY2_CONFIG_MODE | 1;
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(vmrgsort4, dst_ptr, xn, xm, config);
  }
}

__aiv__ __attribute__((always_inline)) void
sort_1024_topk_smallk(memref_t<__ubuf__ float, 1> *src,
                      memref_t<__ubuf__ float, 1> *tmp, int64_t topk,
                      int64_t index_offset,
                      memref_t<__ubuf__ float, 1> *dst_proposals) {
  memref_t<__ubuf__ int32_t, 1> tmp_i32;
  view_as<float, int32_t, 1>(tmp, &tmp_i32);

  memref_t<__ubuf__ int32_t, 1> src_index{tmp_i32.aligned,
                                          tmp_i32.allocated,
                                          tmp_i32.offset + CHUNK * SMALLK_NPP,
                                          {CHUNK},
                                          {1}};
  prepare_index_smallk(&src_index, CHUNK, CHUNK, index_offset);
  prepare_desc_value_smallk(src, CHUNK, CHUNK);

  memref_t<__ubuf__ float, 1> proposals_a{
      tmp->aligned, tmp->allocated, tmp->offset, {CHUNK * SMALLK_NPP}, {1}};
  memref_t<__ubuf__ float, 1> proposals_b{tmp->aligned,
                                          tmp->allocated,
                                          tmp->offset + CHUNK * SMALLK_NPP,
                                          {CHUNK * SMALLK_NPP},
                                          {1}};

  INTRINSIC(pipe_barrier, PIPE_V);
  block_sort_smallk(src, &src_index, &proposals_a, CHUNK);
  INTRINSIC(pipe_barrier, PIPE_V);
  merge_sort_1024_smallk(&proposals_a, &proposals_b);

  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> from{proposals_b.aligned,
                                   proposals_b.allocated,
                                   proposals_b.offset,
                                   {topk * SMALLK_NPP},
                                   {1}};
  memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                 dst_proposals->allocated,
                                 dst_proposals->offset,
                                 {topk * SMALLK_NPP},
                                 {1}};
  vector_eltwise_vs_1d<VectorOpTy::VADDS, float>(&from, 0.0f, &to);
}

extern "C" {

/// Small-K segment topk for SEG_LEN=4096.
///
/// Split one 4096 segment into 4x1024 sub-runs, sort each sub-run and keep K
/// candidates, then 4-way merge the four candidate runs and copy only topK
/// proposals to dst_proposals. This is intended for K <= 1024.
__aiv__ __attribute__((always_inline)) void
sort_s4096_k129_512_impl(memref_t<__ubuf__ float, 1> *src,
                         memref_t<__ubuf__ float, 1> *tmp_buf, bool descending,
                         int64_t topk, int64_t index_offset,
                         memref_t<__ubuf__ float, 1> *dst_proposals) {
  (void)descending;

  int64_t candidates_f32 = WAYS * topk * SMALLK_NPP;
  int64_t sort_tmp_f32 = CHUNK * SMALLK_NPP * 2;
  int64_t sort_tmp_base = tmp_buf->offset + candidates_f32;
  int64_t merge_out_base = sort_tmp_base + sort_tmp_f32;

  for (int64_t lane = 0; lane < WAYS; ++lane) {
    memref_t<__ubuf__ float, 1> src_chunk{
        src->aligned, src->allocated, src->offset + lane * CHUNK, {CHUNK}, {1}};
    memref_t<__ubuf__ float, 1> sort_tmp{tmp_buf->aligned,
                                         tmp_buf->allocated,
                                         sort_tmp_base,
                                         {sort_tmp_f32},
                                         {1}};
    memref_t<__ubuf__ float, 1> chunk_dst{tmp_buf->aligned,
                                          tmp_buf->allocated,
                                          tmp_buf->offset +
                                              lane * topk * SMALLK_NPP,
                                          {topk * SMALLK_NPP},
                                          {1}};
    sort_1024_topk_smallk(&src_chunk, &sort_tmp, topk,
                          index_offset + lane * CHUNK, &chunk_dst);
  }

  auto tmp_ptr = tmp_buf->aligned + tmp_buf->offset;
  __ubuf__ float *xn[WAYS] = {
      tmp_ptr + 0 * topk * SMALLK_NPP,
      tmp_ptr + 1 * topk * SMALLK_NPP,
      tmp_ptr + 2 * topk * SMALLK_NPP,
      tmp_ptr + 3 * topk * SMALLK_NPP,
  };
  uint32_t lens[WAYS] = {
      (uint32_t)topk,
      (uint32_t)topk,
      (uint32_t)topk,
      (uint32_t)topk,
  };

  __ubuf__ float *merge_out = tmp_buf->aligned + merge_out_base;
  uint64_t sr = vmrgsort4_exhaust<float>(merge_out, xn, lens, WAYS);
  int64_t produced = 0;
  for (int i = 0; i < WAYS; ++i) {
    produced += (int64_t)vms4_consumed(sr, i);
  }
  if (produced < topk) {
    // Defensive fallback for unexpected data/configuration.  Valid 4xK input
    // should always produce at least K proposals when K <= 1024.
    topk = produced;
  }

  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> from{tmp_buf->aligned,
                                   tmp_buf->allocated,
                                   merge_out_base,
                                   {topk * SMALLK_NPP},
                                   {1}};
  memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                 dst_proposals->allocated,
                                 dst_proposals->offset,
                                 {topk * SMALLK_NPP},
                                 {1}};
  vector_eltwise_vs_1d<VectorOpTy::VADDS, float>(&from, 0.0f, &to);
}

} // extern "C"
