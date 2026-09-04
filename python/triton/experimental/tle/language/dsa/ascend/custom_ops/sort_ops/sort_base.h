// Copyright 2026- Xcoresigma Technology Co., Ltd

// Included only by custom_ops_aiv.cpp.

#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void check_inputs_of_sort_1d_with_index(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index = nullptr) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;
  assert((src->strides[0] == 1) && "The src must be continues.");
  assert((dst_value->strides[0] == 1) && "The dst_value must be continues.");
  if (dst_index) {
    assert((dst_index->strides[0] == 1) && "The dst_index must be continues.");
  }
#endif
}

template <typename T>
__aiv__ __attribute__((always_inline)) void lower_sort_for_last_axis_is_one(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index, bool need_index) {
  if (need_index) {
    auto dst_index_ptr = dst_index->aligned + dst_index->offset;
    brc_scalar_core_1d(0, dst_index_ptr, dst_index->sizes[0]);
  }
  copy_ubuf_to_ubuf_1d_core(src, dst_value);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
prepare_src_value(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                  int64_t real_num, int64_t sort_num, bool descending) {
  auto src_ptr = src->aligned + src->offset;
  int64_t fill_num = sort_num - real_num;
  // Since the sorting instruction only supports descending sorting, the data
  // needs to be reversed for sorting in ascending order, and then reversed
  // after the sorting is completed. Therefore, the maximum value needs to be
  // filled to ensure that it does not affect the sorting result.
  if (!descending) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < fill_num; ++i) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
      if constexpr (std::is_same<T, half>::value) {
        *(src_ptr + real_num + i) =
            half({static_cast<unsigned short>(FLOAT_NAN)});
      } else {
        *(src_ptr + real_num + i) = static_cast<T>(FLOAT_NAN);
      }
#else
      *(src_ptr + real_num + i) = static_cast<T>(FLOAT_NAN);
#endif
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

    // for ascend sort, we need reverse data, because vbs/vms only support
    // descend.
    src->sizes[0] = sort_num;
    if constexpr (sizeof(T) == 4) {
      memref_t<__ubuf__ int32_t, 1> src_s32;
      view_as<T, int32_t, 1>(src, &src_s32);
      memref_t<__ubuf__ int32_t, 1> dst_s32;
      view_as<T, int32_t, 1>(dst, &dst_s32);
      vector_eltwise_vs_1d<VectorOpTy::VADDS, int32_t>(&src_s32, S32_MIN_VALUE,
                                                       &dst_s32);
    } else {
      memref_t<__ubuf__ int16_t, 1> src_s16;
      view_as<T, int16_t, 1>(src, &src_s16);
      memref_t<__ubuf__ int16_t, 1> dst_s16;
      view_as<T, int16_t, 1>(dst, &dst_s16);
      vector_eltwise_vs_1d<VectorOpTy::VADDS, int16_t>(&src_s16, S16_MIN_VALUE,
                                                       &dst_s16);
    }
    src->sizes[0] = real_num;
    return;
  }
  // In descending order, the minimum value needs to be filled to ensure that
  // it does not affect the sorting result.
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < fill_num; ++i) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    if constexpr (std::is_same<T, half>::value) {
      *(src_ptr + real_num + i) =
          half({static_cast<unsigned short>(FLOAT_NEG_INF)});
    } else {
      *(src_ptr + real_num + i) = static_cast<T>(FLOAT_NEG_INF);
    }
#else
    *(src_ptr + real_num + i) = static_cast<T>(FLOAT_NEG_INF);
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) static void
prepare_src_index(memref_t<__ubuf__ int32_t, 1> *src_index, int64_t real_num,
                  int64_t sort_num, int64_t index_offset = 0) {
  src_index->sizes[0] = sort_num;
  arange_1d(src_index, index_offset, 1);
  src_index->sizes[0] = real_num;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
block_sort(memref_t<__ubuf__ T, 1> *src_value,
           memref_t<__ubuf__ int32_t, 1> *src_index,
           memref_t<__ubuf__ T, 1> *dst, int64_t real_num, int64_t sort_num) {
  auto repeat = sort_num / BIT_SORT_NUM_PER_REPEAT;
  auto sort_num_per_intrinsic = INTR_MAX_REPEAT_CNTS * BIT_SORT_NUM_PER_REPEAT;
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_value_ptr = src_value->aligned + src_value->offset;
  auto src_index_ptr = src_index->aligned + src_index->offset;
  if (repeat >= INTR_MAX_REPEAT_CNTS) [[unlikely]] {
    for (int64_t i = 0; i < repeat / INTR_MAX_REPEAT_CNTS; ++i) {
      INTRINSIC(
          vbitsort, dst_ptr + i * sort_num_per_intrinsic * num_per_proposal,
          src_value_ptr + i * sort_num_per_intrinsic,
          (__ubuf__ uint32_t *)(src_index_ptr + i * sort_num_per_intrinsic),
          INTR_MAX_REPEAT_CNTS); // repeat
    }
  }

  if (repeat % INTR_MAX_REPEAT_CNTS != 0) [[likely]] {
    auto loop_num = repeat / INTR_MAX_REPEAT_CNTS;
    INTRINSIC(vbitsort,
              dst_ptr + loop_num * sort_num_per_intrinsic * num_per_proposal,
              src_value_ptr + loop_num * sort_num_per_intrinsic,
              (__ubuf__ uint32_t *)(src_index_ptr +
                                    loop_num * sort_num_per_intrinsic),
              repeat % INTR_MAX_REPEAT_CNTS); // repeat
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
lower_vms_quotient(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                   int64_t &merge_times, int64_t factor, int64_t sort_num) {
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  int64_t list_interval_offset = factor * num_per_proposal;
  int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  // Xn[15:0] is list 0 address
  // Xn[31:16] is list 1 address
  // Xn[47:32] is list 2 address
  // Xn[63:48] is list 3 address
  __ubuf__ T *xn[4] = {
      merge_times % 2 ? src_ptr : dst_ptr,
      merge_times % 2 ? src_ptr + list_interval_offset
                      : dst_ptr + list_interval_offset,
      merge_times % 2 ? src_ptr + list_interval_offset * 2
                      : dst_ptr + list_interval_offset * 2,
      merge_times % 2 ? src_ptr + list_interval_offset * 3
                      : dst_ptr + list_interval_offset * 3,
  };

  // xm[15:0]: number of structures in input list 0
  // xm[31:16]: number of structures in input list 1
  // xm[47:32]: number of structures in input list 2
  // xm[63:48]: number of structures in input list 3
  uint64_t xm =
      ((list_num | (list_num << 16)) | (list_num << 32)) | (list_num << 48);

  // Enabled repeat mode and disable exhaustion mode when running vmrgsort4
  // (config[12] = 0 and config[7:0] > 1 and config[11:8] means whether the
  // four lists valid?)
  uint64_t config =
      (WAY4_CONFIG_MODE | (quotient_repeat & INTR_MAX_REPEAT_CNTS));
  INTRINSIC(vmrgsort4, merge_times % 2 ? dst_ptr : src_ptr,
            xn,      // the address of each list
            xm,      // the number of each list
            config); // config
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
lower_vms_remainder(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                    int64_t &merge_times, int64_t factor, int64_t sort_num,
                    merge_remainder_info *merge_remainder) {
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
  int64_t cur_remainder = sort_num / factor % MERGE_SORT_MAX_PROPOSALS_LIST;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  // Calculate the offset of cur_remainder
  auto base_offset = quotient_repeat * factor * MERGE_SORT_MAX_PROPOSALS_LIST *
                     num_per_proposal;

  if (cur_remainder == 0 && merge_remainder->merge_remainder_flag) {
    memref_t<__ubuf__ T, 1> src_tmp{
        src->aligned,
        src->allocated,
        static_cast<int64_t>(src->offset + base_offset),
        {static_cast<int64_t>(merge_remainder->merge_remainder_num *
                              num_per_proposal)},
        {1}};
    memref_t<__ubuf__ T, 1> dst_tmp{
        dst->aligned,
        dst->allocated,
        static_cast<int64_t>(dst->offset + base_offset),
        {static_cast<int64_t>(merge_remainder->merge_remainder_num *
                              num_per_proposal)},
        {1}};
    // At this point, there is only one list, no need to merge
    copy_ubuf_to_ubuf_1d_core(merge_times % 2 ? &src_tmp : &dst_tmp,
                              merge_times % 2 ? &dst_tmp : &src_tmp);
  } else if (cur_remainder == 1) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 2-way sort, list1 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[2] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
      };
      // 0011(proposal list) + 000000001(repeat)
      uint64_t config = WAY2_CONFIG_MODE | 1;
      uint64_t xm = list_num |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 16);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      memref_t<__ubuf__ T, 1> src_tmp{
          src->aligned,
          src->allocated,
          static_cast<int64_t>(src->offset + base_offset),
          {static_cast<int64_t>(list_num * num_per_proposal)},
          {1}};
      memref_t<__ubuf__ T, 1> dst_tmp{
          dst->aligned,
          dst->allocated,
          static_cast<int64_t>(dst->offset + base_offset),
          {static_cast<int64_t>(list_num * num_per_proposal)},
          {1}};
      // At this point, there is only one list, no need to merge
      copy_ubuf_to_ubuf_1d_core(merge_times % 2 ? &src_tmp : &dst_tmp,
                                merge_times % 2 ? &dst_tmp : &src_tmp);
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor;
  } else if (cur_remainder == 2) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 3-way sort, list2 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[3] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
      };
      // 0111(proposal list) + 000000001(repeat)
      uint64_t config = WAY3_CONFIG_MODE | 1;
      uint64_t xm = (list_num | (list_num << 16)) |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 32);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      // Perform a 2-way sort
      __ubuf__ T *xn[2] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
      };
      // 0011(proposal list) + 000000001(repeat)
      uint64_t config = WAY2_CONFIG_MODE | 1;
      uint64_t xm = list_num | (list_num << 16);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor * 2;
  } else if (cur_remainder == 3) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 4-way sort, list3 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[4] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 3
              : dst_ptr + base_offset + factor * num_per_proposal * 3,
      };
      // 1111(proposal list) + 000000001(repeat)
      uint64_t config = WAY4_CONFIG_MODE | 1;
      uint64_t xm = ((list_num | (list_num << 16)) | (list_num << 32)) |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 48);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      // Perform a 3-way sort
      __ubuf__ T *xn[3] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
      };
      // 0111(proposal list) + 000000001(repeat)
      uint64_t config = WAY3_CONFIG_MODE | 1;
      uint64_t xm = (list_num | (list_num << 16)) | (list_num << 32);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor * 3;
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
merge_sort(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
           int64_t real_num, int64_t sort_num, int64_t &merge_times) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);

  merge_remainder_info merge_remainder;
  // Whether there is a tail block left in the previous round of merge.
  merge_remainder.merge_remainder_flag = false;
  // The number of proposals to be sorted in the tail block left over from the
  // previous merge round
  merge_remainder.merge_remainder_num = 0;

  // current merge round number
  while (1) {
    int64_t factor = 32 << (merge_times * 2);
    int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
    merge_times++;

    // Processing 4-way merge of main block
    // The main block requires 4 equal lengths, and the length of each path is
    // factor(32 << (i * 2))
    if (quotient_repeat > 0) {
      INTRINSIC(pipe_barrier, PIPE_V);
      lower_vms_quotient(src, dst, merge_times, factor, sort_num);
    }

    // Processing 2/3/4-way merge of tail block
    INTRINSIC(pipe_barrier, PIPE_V);
    lower_vms_remainder(src, dst, merge_times, factor, sort_num,
                        &merge_remainder);

    // merge completion conditions
    if (quotient_repeat == 0 ||
        (quotient_repeat == 1 && merge_remainder.merge_remainder_num == 0)) {
      break;
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
move_out_result(memref_t<__ubuf__ T, 1> *src,
                memref_t<__ubuf__ T, 1> *dst_value,
                memref_t<__ubuf__ int32_t, 1> *dst_index, bool descending,
                bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];

  auto src_ptr = src->aligned + src->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;

  // Step1: separate dst_value/dst_index from the sorted proposal.
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, real_num);
  if constexpr (sizeof(T) == 4) {
    vreducev2_1d_with_pattern_mode<T, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
        src, dst_value);
  } else if constexpr (sizeof(T) == 2) {
    vreducev2_1d_with_pattern_mode<T, PatternMode::INDEX_0_FROM_4_ELEMENTS>(
        src, dst_value);
  }
  if (need_index) {
    memref_t<__ubuf__ int32_t, 1> src_int32;
    view_as<T, int32_t, 1>(src, &src_int32);
    vreducev2_1d_with_pattern_mode<int32_t,
                                   PatternMode::INDEX_1_FROM_2_ELEMENTS>(
        &src_int32, dst_index);
  }
  INTRINSIC_NO_ARGS(set_mask_norm);

  // Step2: If it is in ascending order, the data needs to be reversed to return
  // to the original data.
  if (!descending) {
    INTRINSIC(pipe_barrier, PIPE_V);
    if constexpr (sizeof(T) == 4) {
      memref_t<__ubuf__ int32_t, 1> dst_value_s32;
      view_as<T, int32_t, 1>(dst_value, &dst_value_s32);
      vector_eltwise_vs_1d<VectorOpTy::VADDS, int32_t>(
          &dst_value_s32, S32_MIN_VALUE, &dst_value_s32);
    } else {
      memref_t<__ubuf__ int16_t, 1> dst_value_s16;
      view_as<T, int16_t, 1>(dst_value, &dst_value_s16);
      vector_eltwise_vs_1d<VectorOpTy::VADDS, int16_t>(
          &dst_value_s16, S16_MIN_VALUE, &dst_value_s16);
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void lower_sort_operation(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index, memref_t<__ubuf__ T, 1> *tmp_buf,
    bool descending, bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  if (real_num == 1) {
    // When the sort axis is 1, no need to sort, move out directly
    lower_sort_for_last_axis_is_one(src, dst_value, dst_index, need_index);
    return;
  }

  // Step1: Prepare src index for vbitsort
  prepare_src_index(dst_index, real_num, sort_num);

  // Step2: Prepare src value for vbitsort:
  // 1. Fill the data to be sorted to ensure that the number is a multiple of 32
  // 2. If it is ascending order, need to reverse the data
  memref_t<__ubuf__ T, 1> src_inverse{
      tmp_buf->aligned, tmp_buf->allocated, tmp_buf->offset, {sort_num}, {1}};
  if (!descending) {
    tmp_buf->offset = tmp_buf->offset + sort_num;
  }
  prepare_src_value(src, &src_inverse, real_num, sort_num, descending);

  // Step3: Vector Bitonic Sorter
  // VBS32 will combine index with corresponding score after sorting and output
  // a proposal structure 8B
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  memref_t<__ubuf__ T, 1> tmp_buf_for_block_sort{
      tmp_buf->aligned,
      tmp_buf->allocated,
      tmp_buf->offset,
      {static_cast<int64_t>(sort_num * num_per_proposal)},
      {1}};
  INTRINSIC(pipe_barrier, PIPE_V);
  block_sort(descending ? src : &src_inverse, dst_index,
             &tmp_buf_for_block_sort, real_num, sort_num);

  // Step4: Vector Merge Sorter
  memref_t<__ubuf__ T, 1> tmp_buf_for_merge_sort{
      tmp_buf->aligned,
      tmp_buf->allocated,
      tmp_buf->offset + static_cast<int64_t>(sort_num * num_per_proposal),
      {static_cast<int64_t>(sort_num * num_per_proposal)},
      {1}};
  int64_t merge_times = 0;
  INTRINSIC(pipe_barrier, PIPE_V);
  merge_sort(&tmp_buf_for_block_sort, &tmp_buf_for_merge_sort, real_num,
             sort_num, merge_times);

  INTRINSIC(pipe_barrier, PIPE_V);
  // Step5: Process proposal data and move it to dst_value/dst_index
  move_out_result(merge_times % 2 ? &tmp_buf_for_merge_sort
                                  : &tmp_buf_for_block_sort,
                  dst_value, dst_index, descending, need_index);
}

/// Sort src=(a,) with stride[1,], a is the sorting axis returns the sorted
/// value of dst_value.
///
/// constraint:
/// 1. Make sure the sum of all bufs is less than UB_MAX.
/// 2. The data type to be sorted only supports half/float.
/// 3. src id Continuous 1D.
/// 4. The size of tmp_buf is as follows:
///    a == 1:
///        tmp_buf = 0
///    a != 1:
///        descending = false:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 11
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 6
///        descending = true:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 10
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 5
///
/// \param:
/// descending: descending = true to sort in descending order, descending =
/// false to sort in ascending order.
template <typename T>
__aiv__ __attribute__((always_inline)) void
sort_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
        bool descending, memref_t<__ubuf__ T, 1> *tmp_buf) {
  check_inputs_of_sort_1d_with_index(src, dst);

  static_assert(
      (std::is_same<T, half>::value || std::is_same<T, float>::value) &&
      "Sort unsupport this type");

  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  // When sorting, need to sort with index, so need to allocate a space in
  // tmp_buf to store the index corresponding to src_value
  memref_t<__ubuf__ int32_t, 1> tmp_buf_int32;
  view_as<T, int32_t, 1>(tmp_buf, &tmp_buf_int32);
  memref_t<__ubuf__ int32_t, 1> dst_index{tmp_buf_int32.aligned,
                                          tmp_buf_int32.allocated,
                                          tmp_buf_int32.offset,
                                          {sort_num},
                                          {1}};
  tmp_buf->offset += sort_num * (sizeof(int32_t) / sizeof(T));
  lower_sort_operation(src, dst, &dst_index, tmp_buf, descending, false);
}

/// Sort src=(a,) with stride[1,], a is the sorting axis returns the sorted
/// value of dst_value and the index corresponding to the dst_value.
///
/// constraint:
/// 1. Make sure the sum of all bufs is less than UB_MAX.
/// 2. The data type to be sorted only supports half/float.
/// 3. src id Continuous 1D.
/// 4. The size of tmp_buf is as follows:
///    a == 1:
///        tmp_buf = 0
///    a != 1:
///        descending = false:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 9
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 5
///        descending = true:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 8
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 4
///
/// \param:
/// descending: descending = true to sort in descending order, descending =
/// false to sort in ascending order.
template <typename T>
__aiv__ __attribute__((always_inline)) void
sort_1d_with_index(memref_t<__ubuf__ T, 1> *src,
                   memref_t<__ubuf__ T, 1> *dst_value,
                   memref_t<__ubuf__ int32_t, 1> *dst_index, bool descending,
                   memref_t<__ubuf__ T, 1> *tmp_buf) {
  check_inputs_of_sort_1d_with_index(src, dst_value, dst_index);

  static_assert(
      (std::is_same<T, half>::value || std::is_same<T, float>::value) &&
      "Sort unsupport this type");
  lower_sort_operation(src, dst_value, dst_index, tmp_buf, descending, true);
}

extern "C" {

/// sort_1d_topk_proposals for float32
///
/// Sorts src in descending order, keeps only the top-K proposals in raw
/// proposal format (interleaved [value, index] as float32 pairs), skipping
/// the final unpack (vreducev2) step entirely.
///
/// The output dst_proposals can be directly passed to merge_sort for
/// multi-way merging across tiles, saving the cost of unpacking and
/// re-packing.
///
/// Proposal layout (float32): [val0, idx0_as_f32, val1, idx1_as_f32, ...]
///   - Each proposal occupies 2 x float32 = 8 bytes (PROPOSALS_BYTES)
///   - idx is stored as reinterpret_cast<float>(int32_t index)
///
/// \param src:            tensor<N x f32>         — input values to sort
/// \param dst_proposals:  tensor<topk*2 x f32>   — output top-K proposals
/// (packed)
/// \param descending:     sort direction (true = descending)
/// \param topk:           number of top elements to retain
/// \param tmp_buf:        scratch buffer, size >= aligned(N,32) * 5
/// (descending)
///                        or aligned(N,32) * 6 (ascending)
__aiv__ __attribute__((always_inline)) void
sort_base_impl(memref_t<__ubuf__ float, 1> *src,
               memref_t<__ubuf__ float, 1> *tmp_buf, bool descending,
               int64_t topk, int64_t index_offset,
               memref_t<__ubuf__ float, 1> *dst_proposals) {

  int64_t real_num = src->sizes[0];
  int64_t sort_num = CEIL_FACTOR(real_num, BIT_SORT_NUM_PER_REPEAT);
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(float); // == 2

  // ═══════════════════════════════════════════════════════════════════════
  // Memory layout optimization: reuse src_index space for proposals_b
  //
  // Observation: src_index is only needed during block_sort (Step3).
  // After block_sort completes, src_index is dead — its content has been
  // packed into the proposal format by vbitsort. proposals_b is only
  // needed during merge_sort (Step4). So we can overlap them.
  //
  // Layout (descending, float32):
  //   tmp_buf[0 .. sort_num*2):           proposals_a  (block_sort output)
  //   tmp_buf[sort_num*2 .. sort_num*4):  proposals_b  (merge_sort ping-pong)
  //     └── tmp_buf[sort_num*2 .. sort_num*3): also used as src_index
  //         (safe: src_index is consumed before merge_sort overwrites it)
  //
  // Layout (ascending, float32):
  //   tmp_buf[0 .. sort_num):             src_inverse  (flipped values)
  //   tmp_buf[sort_num .. sort_num*3):    proposals_a
  //   tmp_buf[sort_num*3 .. sort_num*5):  proposals_b
  //     └── tmp_buf[sort_num*3 .. sort_num*4): also used as src_index
  //
  // Total: descending = sort_num * 4, ascending = sort_num * 5
  // ═══════════════════════════════════════════════════════════════════════

  // --- Compute offsets ---
  int64_t base = tmp_buf->offset;
  int64_t proposals_a_offset;
  int64_t proposals_b_offset;
  int64_t src_index_offset;

  if (descending) {
    // [0, sort_num*2): proposals_a
    // [sort_num*2, sort_num*4): proposals_b (front part reused as src_index)
    proposals_a_offset = base;
    proposals_b_offset = base + sort_num * num_per_proposal;
    src_index_offset = proposals_b_offset; // reuse proposals_b's front
  } else {
    // [0, sort_num): src_inverse
    // [sort_num, sort_num*3): proposals_a
    // [sort_num*3, sort_num*5): proposals_b (front part reused as src_index)
    proposals_a_offset = base + sort_num;
    proposals_b_offset = base + sort_num + sort_num * num_per_proposal;
    src_index_offset = proposals_b_offset; // reuse proposals_b's front
  }

  // --- Step1: Prepare index (arange) — placed at proposals_b's start ---
  memref_t<__ubuf__ int32_t, 1> tmp_buf_int32;
  view_as<float, int32_t, 1>(tmp_buf, &tmp_buf_int32);
  // src_index_offset is in float32 units, same as int32 units for float
  memref_t<__ubuf__ int32_t, 1> src_index{tmp_buf_int32.aligned,
                                          tmp_buf_int32.allocated,
                                          src_index_offset,
                                          {sort_num},
                                          {1}};

  prepare_src_index(&src_index, real_num, sort_num, index_offset);

  // --- Step2: Prepare src value (pad + flip for ascending) ---
  if (!descending) {
    memref_t<__ubuf__ float, 1> src_inverse{
        tmp_buf->aligned, tmp_buf->allocated, base, {sort_num}, {1}};
    prepare_src_value(src, &src_inverse, real_num, sort_num, descending);

    // --- Step3: Block sort (vbitsort) → proposal format ---
    memref_t<__ubuf__ float, 1> proposals_a{
        tmp_buf->aligned,
        tmp_buf->allocated,
        proposals_a_offset,
        {static_cast<int64_t>(sort_num * num_per_proposal)},
        {1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    block_sort(&src_inverse, &src_index, &proposals_a, real_num, sort_num);

    // --- Step4: Merge sort (vmrgsort4) ---
    // After block_sort, src_index is dead. proposals_b can now safely use
    // the same memory region.
    memref_t<__ubuf__ float, 1> proposals_b{
        tmp_buf->aligned,
        tmp_buf->allocated,
        proposals_b_offset,
        {static_cast<int64_t>(sort_num * num_per_proposal)},
        {1}};
    int64_t merge_times = 0;
    INTRINSIC(pipe_barrier, PIPE_V);
    merge_sort(&proposals_a, &proposals_b, real_num, sort_num, merge_times);

    // --- Step5: Truncate top-K proposals → dst_proposals (NO unpack) ---
    INTRINSIC(pipe_barrier, PIPE_V);
    int64_t copy_size = topk * num_per_proposal;
    memref_t<__ubuf__ float, 1> src_slice{
        (merge_times % 2) ? proposals_b.aligned : proposals_a.aligned,
        (merge_times % 2) ? proposals_b.allocated : proposals_a.allocated,
        (merge_times % 2) ? proposals_b.offset : proposals_a.offset,
        {copy_size},
        {1}};
    memref_t<__ubuf__ float, 1> dst_slice{dst_proposals->aligned,
                                          dst_proposals->allocated,
                                          dst_proposals->offset,
                                          {copy_size},
                                          {1}};
    copy_ubuf_to_ubuf_1d_core(&src_slice, &dst_slice);
  } else {
    // descending: no src_inverse needed, block_sort reads src directly
    prepare_src_value(src, src, real_num, sort_num, descending);

    // --- Step3: Block sort (vbitsort) → proposal format ---
    memref_t<__ubuf__ float, 1> proposals_a{
        tmp_buf->aligned,
        tmp_buf->allocated,
        proposals_a_offset,
        {static_cast<int64_t>(sort_num * num_per_proposal)},
        {1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    block_sort(src, &src_index, &proposals_a, real_num, sort_num);

    // --- Step4: Merge sort (vmrgsort4) ---
    // After block_sort, src_index is dead. proposals_b can now safely use
    // the same memory region.
    memref_t<__ubuf__ float, 1> proposals_b{
        tmp_buf->aligned,
        tmp_buf->allocated,
        proposals_b_offset,
        {static_cast<int64_t>(sort_num * num_per_proposal)},
        {1}};
    int64_t merge_times = 0;
    INTRINSIC(pipe_barrier, PIPE_V);
    merge_sort(&proposals_a, &proposals_b, real_num, sort_num, merge_times);

    // --- Step5: Truncate top-K proposals → dst_proposals (NO unpack) ---
    INTRINSIC(pipe_barrier, PIPE_V);
    int64_t copy_size = topk * num_per_proposal;
    memref_t<__ubuf__ float, 1> src_slice{
        (merge_times % 2) ? proposals_b.aligned : proposals_a.aligned,
        (merge_times % 2) ? proposals_b.allocated : proposals_a.allocated,
        (merge_times % 2) ? proposals_b.offset : proposals_a.offset,
        {copy_size},
        {1}};
    memref_t<__ubuf__ float, 1> dst_slice{dst_proposals->aligned,
                                          dst_proposals->allocated,
                                          dst_proposals->offset,
                                          {copy_size},
                                          {1}};
    copy_ubuf_to_ubuf_1d_core(&src_slice, &dst_slice);
  }
}

} // extern "C"
