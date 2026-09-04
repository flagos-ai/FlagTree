// Copyright 2026- Xcoresigma Technology Co., Ltd

// Included only by custom_ops_aiv.cpp.
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"
#include "sort_common.h"

constexpr int64_t SEG_LEN = 4096;
constexpr int64_t RUN_LEN = 512;
constexpr int64_t NUM_RUNS = SEG_LEN / RUN_LEN;
constexpr int64_t GROUPS = NUM_RUNS / 4;
constexpr int64_t LAYERED_NPP = PROPOSALS_BYTES / sizeof(float);
constexpr int64_t PROPS_A_F32 = SEG_LEN * LAYERED_NPP;
constexpr int64_t PROPS_B_F32 = SEG_LEN * LAYERED_NPP;
constexpr int64_t GROUP_BUF_PROPS = 4 * RUN_LEN;
constexpr int64_t GROUP_BUF_F32 = GROUP_BUF_PROPS * LAYERED_NPP;
constexpr int64_t FINAL_OUT_PROPS = GROUPS * GROUP_BUF_PROPS;
constexpr int64_t FINAL_OUT_F32 = FINAL_OUT_PROPS * LAYERED_NPP + 8;

__aiv__ __attribute__((always_inline)) void
copy_ub_float(__ubuf__ float *src, int64_t src_off, __ubuf__ float *dst,
              int64_t dst_off, int64_t count) {
  if (count <= 0) {
    return;
  }
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> from{src, src, src_off, {count}, {1}};
  memref_t<__ubuf__ float, 1> to{dst, dst, dst_off, {count}, {1}};
  vector_eltwise_vs_1d<VectorOpTy::VADDS, float>(&from, 0.0f, &to);
}

__aiv__ __attribute__((always_inline)) void
prepare_index_layered(memref_t<__ubuf__ int32_t, 1> *src_index,
                      int64_t real_num, int64_t sort_num,
                      int64_t index_offset) {
  src_index->sizes[0] = sort_num;
  arange_1d(src_index, index_offset, 1);
  src_index->sizes[0] = real_num;
}

__aiv__ __attribute__((always_inline)) void
prepare_desc_value_layered(memref_t<__ubuf__ float, 1> *src, int64_t real_num,
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
block_sort_layered(memref_t<__ubuf__ float, 1> *src_value,
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
          vbitsort, dst_ptr + i * sort_num_per_intrinsic * LAYERED_NPP,
          src_value_ptr + i * sort_num_per_intrinsic,
          (__ubuf__ uint32_t *)(src_index_ptr + i * sort_num_per_intrinsic),
          INTR_MAX_REPEAT_CNTS);
    }
  }

  if (repeat % INTR_MAX_REPEAT_CNTS != 0) {
    int64_t loop_num = repeat / INTR_MAX_REPEAT_CNTS;
    INTRINSIC(vbitsort,
              dst_ptr + loop_num * sort_num_per_intrinsic * LAYERED_NPP,
              src_value_ptr + loop_num * sort_num_per_intrinsic,
              (__ubuf__ uint32_t *)(src_index_ptr +
                                    loop_num * sort_num_per_intrinsic),
              repeat % INTR_MAX_REPEAT_CNTS);
  }
}

__aiv__ __attribute__((always_inline)) void
merge_stage4_layered(memref_t<__ubuf__ float, 1> *src,
                     memref_t<__ubuf__ float, 1> *dst, int64_t factor,
                     int64_t repeat) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  int64_t list_interval_offset = factor * LAYERED_NPP;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  __ubuf__ float *xn[4] = {
      src_ptr,
      src_ptr + list_interval_offset,
      src_ptr + list_interval_offset * 2,
      src_ptr + list_interval_offset * 3,
  };
  uint64_t xm =
      ((list_num | (list_num << 16)) | (list_num << 32)) | (list_num << 48);
  uint64_t config = WAY4_CONFIG_MODE | (repeat & INTR_MAX_REPEAT_CNTS);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmrgsort4, dst_ptr, xn, xm, config);
}

__aiv__ __attribute__((always_inline)) int64_t vms_total_layered(uint64_t sr,
                                                                 int ways) {
  int64_t total = 0;
  for (int i = 0; i < ways; ++i) {
    total += (int64_t)vms4_consumed(sr, i);
  }
  return total;
}

__aiv__ __attribute__((always_inline)) void
merge_stage2_layered(memref_t<__ubuf__ float, 1> *src,
                     memref_t<__ubuf__ float, 1> *dst, int64_t factor,
                     int64_t repeat) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  uint64_t xm = list_num | (list_num << 16);
  uint64_t config = WAY2_CONFIG_MODE | 1;
  for (int64_t r = 0; r < repeat; ++r) {
    __ubuf__ float *xn[2] = {
        src_ptr + (2 * r) * factor * LAYERED_NPP,
        src_ptr + (2 * r + 1) * factor * LAYERED_NPP,
    };
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(vmrgsort4, dst_ptr + (2 * r) * factor * LAYERED_NPP, xn, xm,
              config);
  }
}

__aiv__ __attribute__((always_inline)) void merge_8x512_top512_layered(
    memref_t<__ubuf__ float, 1> *runs, __ubuf__ float *group0,
    __ubuf__ float *group1, __ubuf__ float *final_out,
    memref_t<__ubuf__ float, 1> *dst_proposals, int64_t topk) {
  auto run_ptr = runs->aligned + runs->offset;
  __ubuf__ float *group_bufs[2] = {group0, group1};
  uint32_t group_len[2] = {0, 0};

  for (int g = 0; g < 2; ++g) {
    __ubuf__ float *xn[4] = {
        run_ptr + (g * 4 + 0) * RUN_LEN * LAYERED_NPP,
        run_ptr + (g * 4 + 1) * RUN_LEN * LAYERED_NPP,
        run_ptr + (g * 4 + 2) * RUN_LEN * LAYERED_NPP,
        run_ptr + (g * 4 + 3) * RUN_LEN * LAYERED_NPP,
    };
    uint32_t lens[4] = {(uint32_t)RUN_LEN, (uint32_t)RUN_LEN, (uint32_t)RUN_LEN,
                        (uint32_t)RUN_LEN};
    uint64_t sr = vmrgsort4_exhaust<float>(group_bufs[g], xn, lens, 4);
    group_len[g] = (uint32_t)vms_total_layered(sr, 4);
  }

  __ubuf__ float *xn[2] = {group0, group1};
  uint32_t lens[2] = {group_len[0], group_len[1]};
  uint64_t sr = vmrgsort4_exhaust<float>(final_out, xn, lens, 2);
  int64_t produced = vms_total_layered(sr, 2);
  int64_t copy_props = topk;
  if (copy_props > produced) {
    copy_props = produced;
  }
  copy_ub_float(final_out, 0, dst_proposals->aligned + dst_proposals->offset, 0,
                copy_props * LAYERED_NPP);
}

__aiv__ __attribute__((always_inline)) void merge_4x1024_top1024_layered(
    memref_t<__ubuf__ float, 1> *runs1024, __ubuf__ float *final_out,
    memref_t<__ubuf__ float, 1> *dst_proposals, int64_t topk) {
  auto run_ptr = runs1024->aligned + runs1024->offset;
  __ubuf__ float *xn[4] = {
      run_ptr + 0 * 1024 * LAYERED_NPP,
      run_ptr + 1 * 1024 * LAYERED_NPP,
      run_ptr + 2 * 1024 * LAYERED_NPP,
      run_ptr + 3 * 1024 * LAYERED_NPP,
  };
  uint32_t lens[4] = {1024, 1024, 1024, 1024};
  uint64_t sr = vmrgsort4_exhaust<float>(final_out, xn, lens, 4);
  int64_t produced = vms_total_layered(sr, 4);
  int64_t copy_props = topk;
  if (copy_props > produced) {
    copy_props = produced;
  }
  copy_ub_float(final_out, 0, dst_proposals->aligned + dst_proposals->offset, 0,
                copy_props * LAYERED_NPP);
}

__aiv__ __attribute__((always_inline)) void
merge_groups_topk_layered(__ubuf__ float *src, int64_t src_run_len,
                          int64_t num_runs, __ubuf__ float *dst,
                          int64_t dst_run_len, __ubuf__ float *scratch) {
  int64_t out_run = 0;
  for (int64_t base_run = 0; base_run < num_runs; base_run += 4) {
    __ubuf__ float *xn[4];
    uint32_t lens[4] = {0, 0, 0, 0};
    int active = 0;
    for (int i = 0; i < 4; ++i) {
      int64_t run = base_run + i;
      if (run >= num_runs) {
        break;
      }
      xn[active] = src + run * src_run_len * LAYERED_NPP;
      lens[active] = (uint32_t)src_run_len;
      active++;
    }

    if (active == 1) {
      copy_ub_float(src, base_run * src_run_len * LAYERED_NPP, dst,
                    out_run * dst_run_len * LAYERED_NPP,
                    dst_run_len * LAYERED_NPP);
      out_run++;
      continue;
    }

    uint64_t sr = vmrgsort4_exhaust<float>(scratch, xn, lens, active);
    int64_t produced = vms_total_layered(sr, active);
    int64_t copy_props = dst_run_len;
    if (copy_props > produced) {
      copy_props = produced;
    }
    copy_ub_float(scratch, 0, dst, out_run * dst_run_len * LAYERED_NPP,
                  copy_props * LAYERED_NPP);
    out_run++;
  }
}

__aiv__ __attribute__((always_inline)) void
final_merge_runs_topk_layered(__ubuf__ float *src, int64_t run_len,
                              int64_t num_runs, __ubuf__ float *scratch,
                              memref_t<__ubuf__ float, 1> *dst_proposals,
                              int64_t topk) {
  if (num_runs <= 1) {
    copy_ub_float(src, 0, dst_proposals->aligned + dst_proposals->offset, 0,
                  topk * LAYERED_NPP);
    return;
  }

  __ubuf__ float *xn[4];
  uint32_t lens[4] = {0, 0, 0, 0};
  int active = 0;
  for (int64_t run = 0; run < num_runs && run < 4; ++run) {
    xn[active] = src + run * run_len * LAYERED_NPP;
    lens[active] = (uint32_t)run_len;
    active++;
  }
  uint64_t sr = vmrgsort4_exhaust<float>(scratch, xn, lens, active);
  int64_t produced = vms_total_layered(sr, active);
  int64_t copy_props = topk;
  if (copy_props > produced) {
    copy_props = produced;
  }
  copy_ub_float(scratch, 0, dst_proposals->aligned + dst_proposals->offset, 0,
                copy_props * LAYERED_NPP);
}

__aiv__ __attribute__((always_inline)) void merge_128x32_top32_layered(
    memref_t<__ubuf__ float, 1> *runs32, memref_t<__ubuf__ float, 1> *tmp_runs,
    __ubuf__ float *scratch, memref_t<__ubuf__ float, 1> *dst_proposals,
    int64_t topk) {
  auto a = runs32->aligned + runs32->offset;
  auto b = tmp_runs->aligned + tmp_runs->offset;
  merge_groups_topk_layered(a, 32, 128, b, 32, scratch); // 128 -> 32
  merge_groups_topk_layered(b, 32, 32, a, 32, scratch);  // 32 -> 8
  merge_groups_topk_layered(a, 32, 8, b, 32, scratch);   // 8 -> 2
  final_merge_runs_topk_layered(b, 32, 2, scratch, dst_proposals, topk);
}

__aiv__ __attribute__((always_inline)) void merge_32x128_top128_layered(
    memref_t<__ubuf__ float, 1> *runs128, memref_t<__ubuf__ float, 1> *tmp_runs,
    __ubuf__ float *scratch, memref_t<__ubuf__ float, 1> *dst_proposals,
    int64_t topk) {
  auto a = runs128->aligned + runs128->offset;
  auto b = tmp_runs->aligned + tmp_runs->offset;
  merge_groups_topk_layered(a, 128, 32, b, 128, scratch); // 32 -> 8
  merge_groups_topk_layered(b, 128, 8, a, 128, scratch);  // 8 -> 2
  final_merge_runs_topk_layered(a, 128, 2, scratch, dst_proposals, topk);
}

__aiv__ __attribute__((always_inline)) int64_t
refill_group_layered(memref_t<__ubuf__ float, 1> *runs, int64_t group_id,
                     int64_t *raw_cursor, __ubuf__ float *group_buf) {
  auto run_ptr = runs->aligned + runs->offset;
  __ubuf__ float *xn[4];
  uint32_t lens[4] = {0, 0, 0, 0};
  int orig_idx[4] = {0, 0, 0, 0};
  int active = 0;

  int64_t base_run = group_id * 4;
  for (int i = 0; i < 4; ++i) {
    int64_t run = base_run + i;
    int64_t remain = RUN_LEN - raw_cursor[run];
    if (remain <= 0) {
      continue;
    }
    xn[active] = run_ptr + (run * RUN_LEN + raw_cursor[run]) * LAYERED_NPP;
    lens[active] = (uint32_t)remain;
    orig_idx[active] = i;
    active++;
  }

  if (active == 0) {
    return 0;
  }

  if (active == 1) {
    int64_t run = base_run + orig_idx[0];
    int64_t take = RUN_LEN - raw_cursor[run];
    copy_ub_float(run_ptr, (run * RUN_LEN + raw_cursor[run]) * LAYERED_NPP,
                  group_buf, 0, take * LAYERED_NPP);
    raw_cursor[run] += take;
    return take;
  }

  uint64_t sr = vmrgsort4_exhaust<float>(group_buf, xn, lens, active);
  int64_t produced = 0;
  for (int w = 0; w < active; ++w) {
    int64_t consumed = (int64_t)vms4_consumed(sr, w);
    int64_t run = base_run + orig_idx[w];
    raw_cursor[run] += consumed;
    produced += consumed;
  }
  return produced;
}

__aiv__ __attribute__((always_inline)) void merge_8x512_topk_layered(
    memref_t<__ubuf__ float, 1> *runs, __ubuf__ float *group0,
    __ubuf__ float *group1, __ubuf__ float *final_out,
    memref_t<__ubuf__ float, 1> *dst_proposals, int64_t topk) {
  __ubuf__ float *group_bufs[2] = {group0, group1};
  int64_t raw_cursor[NUM_RUNS];
  int64_t group_pos[GROUPS] = {0, 0};
  int64_t group_len[GROUPS] = {0, 0};
  for (int i = 0; i < NUM_RUNS; ++i) {
    raw_cursor[i] = 0;
  }

  for (int g = 0; g < GROUPS; ++g) {
    group_len[g] = refill_group_layered(runs, g, raw_cursor, group_bufs[g]);
    group_pos[g] = 0;
  }

  auto dst_ptr = dst_proposals->aligned + dst_proposals->offset;
  int64_t produced_total = 0;
  while (produced_total < topk) {
    for (int g = 0; g < GROUPS; ++g) {
      if (group_pos[g] >= group_len[g]) {
        group_len[g] = refill_group_layered(runs, g, raw_cursor, group_bufs[g]);
        group_pos[g] = 0;
      }
    }

    __ubuf__ float *xn[2];
    uint32_t lens[2] = {0, 0};
    int orig_idx[2] = {0, 0};
    int active = 0;
    for (int g = 0; g < GROUPS; ++g) {
      int64_t remain = group_len[g] - group_pos[g];
      if (remain <= 0) {
        continue;
      }
      xn[active] = group_bufs[g] + group_pos[g] * LAYERED_NPP;
      lens[active] = (uint32_t)remain;
      orig_idx[active] = g;
      active++;
    }

    if (active == 0) {
      break;
    }

    if (active == 1) {
      int g = orig_idx[0];
      int64_t take = group_len[g] - group_pos[g];
      if (produced_total + take > topk) {
        take = topk - produced_total;
      }
      copy_ub_float(group_bufs[g], group_pos[g] * LAYERED_NPP, dst_ptr,
                    produced_total * LAYERED_NPP, take * LAYERED_NPP);
      group_pos[g] += take;
      produced_total += take;
      continue;
    }

    uint64_t sr = vmrgsort4_exhaust<float>(final_out, xn, lens, active);
    int64_t batch = 0;
    int64_t consumed_by_group[GROUPS] = {0, 0};
    for (int w = 0; w < active; ++w) {
      int64_t consumed = (int64_t)vms4_consumed(sr, w);
      consumed_by_group[orig_idx[w]] = consumed;
      batch += consumed;
    }

    if (batch <= 0) {
      break;
    }

    int64_t copy_props = batch;
    if (produced_total + copy_props > topk) {
      copy_props = topk - produced_total;
    }
    copy_ub_float(final_out, 0, dst_ptr, produced_total * LAYERED_NPP,
                  copy_props * LAYERED_NPP);

    for (int g = 0; g < GROUPS; ++g) {
      group_pos[g] += consumed_by_group[g];
    }
    produced_total += copy_props;
  }
}

extern "C" {

/// Layered TopK sort for one 4096-f32 segment.
///
/// The normal custom_sort fully merges 4096 proposals, then copies topK.  This
/// variant stops after building eight sorted 512-proposal runs and streams a
/// truncated merge from those runs.  It is intended for K <= 1024.
__aiv__ __attribute__((always_inline)) void sort_s4096_k1_128_k2048_impl(
    memref_t<__ubuf__ float, 1> *src, memref_t<__ubuf__ float, 1> *tmp_buf,
    bool descending, int64_t topk, int64_t index_offset,
    memref_t<__ubuf__ float, 1> *dst_proposals) {
  (void)descending;

  int64_t real_num = src->sizes[0];
  int64_t sort_num = SEG_LEN;
  if (topk > 2048) {
    topk = 2048;
  }

  int64_t base = tmp_buf->offset;
  int64_t proposals_a_offset = base;
  int64_t proposals_b_offset = base + PROPS_A_F32;
  int64_t group0_offset = proposals_b_offset + PROPS_B_F32;
  int64_t group1_offset = group0_offset + GROUP_BUF_F32;
  int64_t final_out_offset = group1_offset + GROUP_BUF_F32;

  memref_t<__ubuf__ float, 1> proposals_a{tmp_buf->aligned,
                                          tmp_buf->allocated,
                                          proposals_a_offset,
                                          {PROPS_A_F32},
                                          {1}};
  memref_t<__ubuf__ float, 1> proposals_b{tmp_buf->aligned,
                                          tmp_buf->allocated,
                                          proposals_b_offset,
                                          {PROPS_B_F32},
                                          {1}};

  memref_t<__ubuf__ int32_t, 1> tmp_i32;
  view_as<float, int32_t, 1>(tmp_buf, &tmp_i32);
  memref_t<__ubuf__ int32_t, 1> src_index{
      tmp_i32.aligned, tmp_i32.allocated, proposals_b_offset, {SEG_LEN}, {1}};

  prepare_index_layered(&src_index, real_num, sort_num, index_offset);
  prepare_desc_value_layered(src, real_num, sort_num);

  INTRINSIC(pipe_barrier, PIPE_V);
  block_sort_layered(src, &src_index, &proposals_a, sort_num);

  __ubuf__ float *group0 = tmp_buf->aligned + group0_offset;
  __ubuf__ float *group1 = tmp_buf->aligned + group1_offset;
  __ubuf__ float *final_out = tmp_buf->aligned + final_out_offset;

  if (topk <= 32) {
    merge_128x32_top32_layered(&proposals_a, &proposals_b, final_out,
                               dst_proposals, topk);
    return;
  }

  // 128x32 VBS runs -> 32x128 runs.
  merge_stage4_layered(&proposals_a, &proposals_b, 32, 32);

  if (topk <= 128) {
    merge_32x128_top128_layered(&proposals_b, &proposals_a, final_out,
                                dst_proposals, topk);
    return;
  }

  // 32x128 runs -> 8x512 runs.
  merge_stage4_layered(&proposals_b, &proposals_a, 128, 8);

  if (topk <= 512) {
    merge_8x512_top512_layered(&proposals_a, group0, group1, final_out,
                               dst_proposals, topk);
  } else {
    // 8x512 -> 2x2048, then one truncated 2-way merge.  TopK<=2048 can never
    // need more than 2048 proposals from either 4-run group.
    merge_stage4_layered(&proposals_a, &proposals_b, 512, 2);
    final_merge_runs_topk_layered(proposals_b.aligned + proposals_b.offset,
                                  2048, 2, final_out, dst_proposals, topk);
  }
}

} // extern "C"
