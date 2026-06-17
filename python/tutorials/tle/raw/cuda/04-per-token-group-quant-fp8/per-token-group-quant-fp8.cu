#include "vectorization.cuh"
#include "vectorization_utils.cuh"
#include <cmath>
#include <cuda_fp8.h>

__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float
ComputeGroupScale(const float *__restrict__ group_input,
                  float *__restrict__ smem_group, const int group_size,
                  const int lane_id, const int threads_per_group,
                  const float eps, const float max_8bit) {
  float local_absmax = eps;

  constexpr int vec_size = 16 / sizeof(float);

  // copy global -> shared & compute absmax
  auto scalar_op_cache = [&] __device__(float &dst, const float &src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(group_input, // in
                                           smem_group,  // out (shared)
                                           group_size,  // elements per group
                                           lane_id,     // thread id
                                           threads_per_group, // stride in group
                                           scalar_op_cache);  // scalar handler

  local_absmax = GroupReduceMax(local_absmax);

  float y_s = local_absmax / max_8bit;
  // if constexpr (SCALE_UE8M0) {
  //   y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  // }

  return y_s;
}

// template <typename T, typename DST_DTYPE>
__device__ __forceinline__ void
QuantizeGroup(const float *__restrict__ smem_group,
              __nv_fp8_e4m3 *__restrict__ group_output, const int group_size,
              const int lane_id, const int threads_per_group, const float y_s,
              const float min_8bit, const float max_8bit) {
  constexpr int vec_size = 16 / sizeof(float);

  // quantize shared -> global 8-bit
  auto scalar_op_quant = [&] __device__(__nv_fp8_e4m3 & dst, const float &src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = __nv_fp8_e4m3(q);
  };

  vllm::vectorize_with_alignment<vec_size>(
      smem_group,        // in (shared)
      group_output,      // out (global quant tensor)
      group_size,        // elements
      lane_id,           // tid
      threads_per_group, // stride
      scalar_op_quant);  // scalar handler
}

// T: float;  DST_DTYPE: __nv_fp8_e4m3
// template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
//           bool SCALE_UE8M0 = false, typename scale_packed_t = float>
// __global__ void per_token_group_quant_8bit_kernel(
extern "C" __device__ void per_token_group_quant_8bit(
    const float *__restrict__ input, void *__restrict__ output_q,
    float *__restrict__ output_s, const int group_size, const int num_groups,
    const int groups_per_block, const float eps, const float min_8bit,
    const float max_8bit) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  static_assert(sizeof(float) % sizeof(float) == 0);

  const float *group_input = input + block_group_offset;
  __nv_fp8_e4m3 *group_output =
      static_cast<__nv_fp8_e4m3 *>(output_q) + block_group_offset;
  float *scale_output;

  // bool IS_COLUMN_MAJOR = false;
  // if (IS_COLUMN_MAJOR) {
  //   const int num_elems_per_pack =
  //       static_cast<int>(sizeof(float) / sizeof(float));
  //   const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
  //   const int row_idx = global_group_id / scale_num_rows_element;
  //   const int col_idx_raw = global_group_id % scale_num_rows_element;
  //   const int col_idx = col_idx_raw / num_elems_per_pack;
  //   const int pack_idx = col_idx_raw % num_elems_per_pack;
  //   scale_output = reinterpret_cast<float*>(output_s) +
  //                  (col_idx * scale_stride * num_elems_per_pack +
  //                   row_idx * num_elems_per_pack + pack_idx);
  // } else {
  scale_output = output_s + global_group_id;
  // }

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  float *smem = reinterpret_cast<float *>(smem_raw);
  float *smem_group = smem + local_group_id * group_size;

  const float y_s =
      ComputeGroupScale(group_input, smem_group, group_size, lane_id,
                        threads_per_group, eps, max_8bit);

  float y_s_quant = y_s;

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  __syncthreads();

  QuantizeGroup(smem_group, group_output, group_size, lane_id,
                threads_per_group, y_s, min_8bit, max_8bit);
}
