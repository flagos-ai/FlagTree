#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <nvshmem.h>

namespace {

constexpr int kNumExpertsLimit = 16;
constexpr int kMaxDispatchWarps = 8;
constexpr uint64_t kNumBarrierSignalBytes = 32;
constexpr int kLCMCandidateBlockM = 384;
constexpr int kMaxCandidateBlockM = 192;
constexpr int kMinCandidateBlockM = 8;
constexpr int kBlockM = 64;

__device__ __forceinline__ uint64_t align_u64(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

__device__ __forceinline__ uint32_t align_u32(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

__device__ __forceinline__ uint32_t num_max_pool_tokens(
    uint32_t num_ranks,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t num_experts_per_rank) {
  const uint32_t num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
  const uint32_t num_max_experts_per_token =
      num_topk < num_experts_per_rank ? num_topk : num_experts_per_rank;
  return align_u32(
      num_max_recv_tokens * num_max_experts_per_token +
          num_experts_per_rank * (kMaxCandidateBlockM - 1),
      kLCMCandidateBlockM);
}

__device__ __forceinline__ uint32_t num_max_pool_blocks(
    uint32_t num_ranks,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t num_experts_per_rank) {
  return num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk,
                             num_experts_per_rank) /
         kMinCandidateBlockM;
}

__device__ __forceinline__ uint64_t workspace_bytes(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk) {
  const uint32_t num_experts_per_rank = num_experts / num_ranks;
  const uint32_t max_recv = num_ranks * num_max_tokens_per_rank;
  const uint32_t pool_tokens =
      num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk,
                          num_experts_per_rank);
  const uint32_t pool_blocks =
      num_max_pool_blocks(num_ranks, num_max_tokens_per_rank, num_topk,
                          num_experts_per_rank);

  uint64_t bytes = 0;
  bytes += kNumBarrierSignalBytes;
  bytes += static_cast<uint64_t>(num_experts) * sizeof(uint64_t) * 2;
  bytes += static_cast<uint64_t>(num_experts_per_rank) * sizeof(uint64_t);
  bytes += static_cast<uint64_t>(align_u32(pool_blocks, 2)) * sizeof(uint32_t);
  bytes += static_cast<uint64_t>(pool_blocks) * sizeof(uint64_t);
  bytes += static_cast<uint64_t>(num_experts_per_rank) * num_ranks * max_recv *
           sizeof(uint32_t);
  bytes += static_cast<uint64_t>(pool_tokens) * 3 * sizeof(uint32_t);
  return align_u64(bytes, 16);
}

__device__ __forceinline__ uint64_t *expert_send_count_ptr(
    uint8_t *base,
    uint32_t expert_idx) {
  return reinterpret_cast<uint64_t *>(base + kNumBarrierSignalBytes) + expert_idx;
}

__device__ __forceinline__ uint64_t *expert_recv_count_ptr(
    uint8_t *base,
    uint32_t num_experts,
    uint32_t num_experts_per_rank,
    uint32_t rank_idx,
    uint32_t expert_idx) {
  return expert_send_count_ptr(base, num_experts) +
         rank_idx * num_experts_per_rank + expert_idx;
}

__device__ __forceinline__ uint64_t *expert_recv_count_sum_ptr(
    uint8_t *base,
    uint32_t num_experts,
    uint32_t num_experts_per_rank,
    uint32_t expert_idx) {
  return expert_send_count_ptr(base, num_experts * 2) + expert_idx;
}

__device__ __forceinline__ uint32_t *l1_arrival_count_ptr(
    uint8_t *base,
    uint32_t num_experts,
    uint32_t num_experts_per_rank,
    uint32_t pool_block_idx) {
  return reinterpret_cast<uint32_t *>(
             expert_recv_count_sum_ptr(base, num_experts, num_experts_per_rank,
                                       num_experts_per_rank)) +
         pool_block_idx;
}

__device__ __forceinline__ uint64_t *l2_arrival_mask_ptr(
    uint8_t *base,
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t pool_block_idx) {
  const uint32_t num_experts_per_rank = num_experts / num_ranks;
  const uint32_t pool_blocks =
      num_max_pool_blocks(num_ranks, num_max_tokens_per_rank, num_topk,
                          num_experts_per_rank);
  return reinterpret_cast<uint64_t *>(
             l1_arrival_count_ptr(base, num_experts, num_experts_per_rank,
                                  align_u32(pool_blocks, 2))) +
         pool_block_idx;
}

__device__ __forceinline__ uint32_t *src_token_topk_idx_ptr(
    uint8_t *base,
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t expert_idx,
    uint32_t rank_idx,
    uint32_t token_idx) {
  const uint32_t num_experts_per_rank = num_experts / num_ranks;
  const uint32_t max_recv = num_ranks * num_max_tokens_per_rank;
  const uint32_t pool_blocks =
      num_max_pool_blocks(num_ranks, num_max_tokens_per_rank, num_topk,
                          num_experts_per_rank);
  return reinterpret_cast<uint32_t *>(
             l2_arrival_mask_ptr(base, num_ranks, num_experts,
                                 num_max_tokens_per_rank, num_topk,
                                 pool_blocks)) +
         expert_idx * (num_ranks * max_recv) + rank_idx * max_recv + token_idx;
}

__device__ __forceinline__ uint32_t *token_src_metadata_ptr(
    uint8_t *base,
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t pool_token_idx) {
  const uint32_t num_experts_per_rank = num_experts / num_ranks;
  return reinterpret_cast<uint32_t *>(src_token_topk_idx_ptr(
             base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
             num_experts_per_rank, 0, 0)) +
         pool_token_idx * 3;
}

__device__ __forceinline__ uint64_t input_token_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk) {
  return workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk);
}

__device__ __forceinline__ uint64_t input_sf_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden) {
  return input_token_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                            num_topk) +
         static_cast<uint64_t>(num_max_tokens_per_rank) * hidden;
}

__device__ __forceinline__ uint64_t input_topk_idx_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden) {
  return input_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk, hidden) +
         static_cast<uint64_t>(num_max_tokens_per_rank) * (hidden / 32);
}

__device__ __forceinline__ uint64_t input_topk_weight_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden) {
  return input_topk_idx_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                               num_topk, hidden) +
         static_cast<uint64_t>(num_max_tokens_per_rank) * num_topk *
             sizeof(int64_t);
}

__device__ __forceinline__ uint64_t l1_token_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden) {
  return input_topk_weight_offset(num_ranks, num_experts,
                                  num_max_tokens_per_rank, num_topk, hidden) +
         static_cast<uint64_t>(num_max_tokens_per_rank) * num_topk *
             sizeof(float);
}

__device__ __forceinline__ uint64_t l1_sf_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden) {
  const uint32_t epr = num_experts / num_ranks;
  const uint32_t pool_tokens =
      num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr);
  return l1_token_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk, hidden) +
         static_cast<uint64_t>(pool_tokens) * hidden;
}

__device__ __forceinline__ uint64_t l1_topk_weight_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden,
    uint32_t num_padded_sf_pool_tokens) {
  return l1_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden) +
         static_cast<uint64_t>(num_padded_sf_pool_tokens) * (hidden / 32);
}

__device__ __forceinline__ uint64_t l2_token_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden,
    uint32_t num_padded_sf_pool_tokens) {
  const uint32_t epr = num_experts / num_ranks;
  const uint32_t pool_tokens =
      num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr);
  return l1_topk_weight_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                               num_topk, hidden,
                               num_padded_sf_pool_tokens) +
         static_cast<uint64_t>(pool_tokens) * sizeof(float);
}

__device__ __forceinline__ uint64_t l2_sf_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden,
    uint32_t intermediate_hidden,
    uint32_t num_padded_sf_pool_tokens) {
  const uint32_t epr = num_experts / num_ranks;
  const uint32_t pool_tokens =
      num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr);
  return l2_token_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk, hidden, num_padded_sf_pool_tokens) +
         static_cast<uint64_t>(pool_tokens) * intermediate_hidden;
}

__device__ __forceinline__ uint64_t combine_token_offset(
    uint32_t num_ranks,
    uint32_t num_experts,
    uint32_t num_max_tokens_per_rank,
    uint32_t num_topk,
    uint32_t hidden,
    uint32_t intermediate_hidden,
    uint32_t num_padded_sf_pool_tokens) {
  return l2_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden, intermediate_hidden,
                      num_padded_sf_pool_tokens) +
         static_cast<uint64_t>(num_padded_sf_pool_tokens) *
             (intermediate_hidden / 16u);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t value) {
  if (value == 0) {
    return 0.0f;
  }
  const int sign = (value & 0x80u) ? -1 : 1;
  const int exp = (value >> 3) & 0x0f;
  const int mant = value & 0x07;
  float mag;
  if (exp == 0) {
    mag = ldexpf(static_cast<float>(mant) / 8.0f, -6);
  } else {
    mag = ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
  }
  return sign < 0 ? -mag : mag;
}

__device__ __forceinline__ void choose_rank_round_robin(
    uint32_t token_idx_in_expert,
    const uint32_t *rank_counts,
    uint32_t num_ranks,
    uint32_t *rank_out,
    uint32_t *token_idx_in_rank_out) {
  uint32_t remaining[8];
  for (uint32_t r = 0; r < num_ranks; ++r) {
    remaining[r] = rank_counts[r];
  }

  uint32_t offset = 0;
  uint32_t slot_idx = token_idx_in_expert;
  while (true) {
    uint32_t active = 0;
    uint32_t length = 0xffffffffu;
    for (uint32_t r = 0; r < num_ranks; ++r) {
      if (remaining[r] > 0) {
        ++active;
        length = remaining[r] < length ? remaining[r] : length;
      }
    }
    if (active == 0) {
      break;
    }
    const uint32_t num_round_tokens = length * active;
    if (slot_idx < num_round_tokens) {
      const uint32_t slot_idx_in_round = slot_idx % active;
      uint32_t seen = 0;
      for (uint32_t r = 0; r < num_ranks; ++r) {
        if (remaining[r] == 0) {
          continue;
        }
        if (slot_idx_in_round == seen) {
          *rank_out = r;
          *token_idx_in_rank_out = offset + slot_idx / active;
          return;
        }
        ++seen;
      }
    }
    slot_idx -= num_round_tokens;
    offset += length;
    for (uint32_t r = 0; r < num_ranks; ++r) {
      remaining[r] -= remaining[r] < length ? remaining[r] : length;
    }
  }
  *rank_out = 0;
  *token_idx_in_rank_out = 0;
}

}  // namespace

extern "C" __device__ void userhopper_ws_dispatch_partition(
    uint8_t *symm_buffer,
    int num_tokens,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden) {
  if (blockIdx.x != 0) {
    return;
  }

  const uint32_t rank = static_cast<uint32_t>(nvshmem_my_pe());
  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  const uint32_t lane_idx = static_cast<uint32_t>(threadIdx.x & 31);
  const uint32_t tokens_per_warp = 32u / static_cast<uint32_t>(num_topk);
  const uint32_t activate_lanes = tokens_per_warp * static_cast<uint32_t>(num_topk);
  __shared__ uint32_t local_counts[kNumExpertsLimit];
  __shared__ uint32_t local_slots[kNumExpertsLimit];
  for (uint32_t i = lane_idx; i < kNumExpertsLimit; i += 32) {
    local_counts[i] = 0;
    local_slots[i] = 0;
  }
  __syncwarp();

  int64_t *topk_idx = reinterpret_cast<int64_t *>(
      symm_buffer + input_topk_idx_offset(num_ranks, num_experts,
                                          num_max_tokens_per_rank, num_topk,
                                          hidden));

  for (uint32_t i = 0; i < static_cast<uint32_t>(num_tokens);
       i += tokens_per_warp) {
    int expert = -1;
    const uint32_t token_idx = i + lane_idx / static_cast<uint32_t>(num_topk);
    const bool active =
        lane_idx < activate_lanes && token_idx < static_cast<uint32_t>(num_tokens);
    const uint32_t token_topk_idx = i * static_cast<uint32_t>(num_topk) + lane_idx;
    if (active) {
      expert = static_cast<int>(topk_idx[token_topk_idx]);
      if (expert >= 0 && expert < num_experts && expert < kNumExpertsLimit) {
        atomicAdd(local_counts + expert, 1u);
      }
    }
    __syncwarp();
  }

  __syncwarp();
  for (uint32_t expert = lane_idx;
       expert < static_cast<uint32_t>(num_experts) && expert < kNumExpertsLimit;
       expert += 32) {
    const uint64_t send_value =
        (static_cast<uint64_t>(1) << 32) | local_counts[expert];
    local_slots[expert] = static_cast<uint32_t>(
        nvshmem_uint64_atomic_fetch_add(expert_send_count_ptr(symm_buffer, expert),
                                        send_value, rank));
  }
  __syncwarp();

  for (uint32_t i = 0; i < static_cast<uint32_t>(num_tokens);
       i += tokens_per_warp) {
    int expert_idx = -1;
    const uint32_t token_idx = i + lane_idx / static_cast<uint32_t>(num_topk);
    const bool active =
        lane_idx < activate_lanes && token_idx < static_cast<uint32_t>(num_tokens);
    const uint32_t token_topk_idx = i * static_cast<uint32_t>(num_topk) + lane_idx;
    if (active) {
      expert_idx = static_cast<int>(topk_idx[token_topk_idx]);
    }
    if (expert_idx >= 0 && expert_idx < num_experts &&
        expert_idx < kNumExpertsLimit) {
      const uint32_t expert = static_cast<uint32_t>(expert_idx);
      const uint32_t dst_rank = expert / epr;
      const uint32_t dst_local_expert = expert - dst_rank * epr;
      const uint32_t dst_slot = atomicAdd(local_slots + expert, 1u);
      uint32_t *remote_queue = src_token_topk_idx_ptr(
          symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
          dst_local_expert, rank, dst_slot);
      nvshmem_uint_p(remote_queue, token_topk_idx, dst_rank);
    }
    __syncwarp();
  }

  __syncwarp();
  nvshmem_quiet();

  for (uint32_t expert = lane_idx;
       expert < static_cast<uint32_t>(num_experts) && expert < kNumExpertsLimit;
       expert += 32) {
    const uint64_t expert_status = *expert_send_count_ptr(symm_buffer, expert);
    const uint32_t dst_rank = expert / epr;
    const uint32_t dst_local_expert = expert - dst_rank * epr;
    nvshmem_uint64_p(
        expert_recv_count_ptr(symm_buffer, num_experts, epr, rank,
                              dst_local_expert),
        expert_status & 0xffffffffu, dst_rank);
    nvshmem_quiet();
    nvshmem_uint64_atomic_add(
        expert_recv_count_sum_ptr(symm_buffer, num_experts, epr,
                                  dst_local_expert),
        expert_status, dst_rank);
  }
  __syncwarp();
  nvshmem_quiet();
}

extern "C" __device__ void userhopper_ws_dispatch_partition_cta_warp0(
    uint8_t *symm_buffer,
    int num_tokens,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden) {
  if (blockIdx.x != 0 || (threadIdx.x >> 5) != 0) {
    return;
  }
  userhopper_ws_dispatch_partition(symm_buffer, num_tokens, num_ranks,
                                   num_experts, num_max_tokens_per_rank,
                                   num_topk, hidden);
}

extern "C" __device__ void userhopper_ws_dispatch_partition_cta_multiwarp(
    uint8_t *symm_buffer,
    int num_tokens,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int num_dispatch_warps) {
  if (blockIdx.x != 0) {
    return;
  }
  const uint32_t warp_idx = static_cast<uint32_t>(threadIdx.x >> 5);
  const uint32_t lane_idx = static_cast<uint32_t>(threadIdx.x & 31);
  uint32_t active_dispatch_warps = static_cast<uint32_t>(num_dispatch_warps);
  active_dispatch_warps =
      active_dispatch_warps < static_cast<uint32_t>(kMaxDispatchWarps)
          ? active_dispatch_warps
          : static_cast<uint32_t>(kMaxDispatchWarps);
  if (warp_idx >= active_dispatch_warps) {
    return;
  }

  const uint32_t rank = static_cast<uint32_t>(nvshmem_my_pe());
  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  const uint32_t tokens_per_warp = 32u / static_cast<uint32_t>(num_topk);
  const uint32_t activate_lanes = tokens_per_warp * static_cast<uint32_t>(num_topk);
  __shared__ uint32_t local_counts[kNumExpertsLimit];
  __shared__ uint32_t local_slots[kNumExpertsLimit];
  __shared__ uint32_t round_counts[kMaxDispatchWarps * kNumExpertsLimit];
  __shared__ uint32_t round_slots[kMaxDispatchWarps * kNumExpertsLimit];
  __shared__ volatile uint32_t sync_state[6];

  if (warp_idx == 0) {
    for (uint32_t i = lane_idx; i < kNumExpertsLimit; i += 32) {
      local_counts[i] = 0;
      local_slots[i] = 0;
    }
    for (uint32_t i = lane_idx;
         i < static_cast<uint32_t>(kMaxDispatchWarps * kNumExpertsLimit);
         i += 32) {
      round_counts[i] = 0;
      round_slots[i] = 0;
    }
    if (lane_idx < 6u) {
      sync_state[lane_idx] = 0;
    }
    __syncwarp();
    if (lane_idx == 0) {
      sync_state[0] = 1;
    }
  } else {
    while (sync_state[0] == 0) {
    }
  }
  __syncwarp();

  int64_t *topk_idx = reinterpret_cast<int64_t *>(
      symm_buffer + input_topk_idx_offset(num_ranks, num_experts,
                                          num_max_tokens_per_rank, num_topk,
                                          hidden));

  const uint32_t stride =
      active_dispatch_warps * tokens_per_warp;
  for (uint32_t i = warp_idx * tokens_per_warp;
       i < static_cast<uint32_t>(num_tokens); i += stride) {
    int expert = -1;
    const uint32_t token_idx = i + lane_idx / static_cast<uint32_t>(num_topk);
    const bool active =
        lane_idx < activate_lanes && token_idx < static_cast<uint32_t>(num_tokens);
    const uint32_t token_topk_idx = i * static_cast<uint32_t>(num_topk) + lane_idx;
    if (active) {
      expert = static_cast<int>(topk_idx[token_topk_idx]);
      if (expert >= 0 && expert < num_experts && expert < kNumExpertsLimit) {
        atomicAdd(local_counts + expert, 1u);
      }
    }
    __syncwarp();
  }
  if (lane_idx == 0) {
    atomicAdd(const_cast<uint32_t *>(&sync_state[1]), 1u);
  }
  while (sync_state[1] < active_dispatch_warps) {
  }
  __syncwarp();

  if (warp_idx == 0) {
    for (uint32_t expert = lane_idx;
         expert < static_cast<uint32_t>(num_experts) && expert < kNumExpertsLimit;
         expert += 32) {
      const uint64_t send_value =
          (static_cast<uint64_t>(1) << 32) | local_counts[expert];
      local_slots[expert] = static_cast<uint32_t>(
          nvshmem_uint64_atomic_fetch_add(expert_send_count_ptr(symm_buffer, expert),
                                          send_value, rank));
    }
    __syncwarp();
    if (lane_idx == 0) {
      sync_state[2] = 1;
    }
  } else {
    while (sync_state[2] == 0) {
    }
  }
  __syncwarp();

  uint32_t round_idx = 0;
  for (uint32_t round_base = 0; round_base < static_cast<uint32_t>(num_tokens);
       round_base += stride, ++round_idx) {
    uint32_t *my_round_counts =
        round_counts + warp_idx * static_cast<uint32_t>(kNumExpertsLimit);
    uint32_t *my_round_slots =
        round_slots + warp_idx * static_cast<uint32_t>(kNumExpertsLimit);
    for (uint32_t expert = lane_idx; expert < kNumExpertsLimit; expert += 32) {
      my_round_counts[expert] = 0;
      my_round_slots[expert] = 0;
    }
    __syncwarp();

    const uint32_t i = round_base + warp_idx * tokens_per_warp;
    int expert_idx = -1;
    const uint32_t token_idx = i + lane_idx / static_cast<uint32_t>(num_topk);
    const bool active =
        lane_idx < activate_lanes && token_idx < static_cast<uint32_t>(num_tokens);
    const uint32_t token_topk_idx = i * static_cast<uint32_t>(num_topk) + lane_idx;
    if (active) {
      expert_idx = static_cast<int>(topk_idx[token_topk_idx]);
      if (expert_idx >= 0 && expert_idx < num_experts &&
          expert_idx < kNumExpertsLimit) {
        atomicAdd(my_round_counts + static_cast<uint32_t>(expert_idx), 1u);
      }
    }
    __syncwarp();

    const uint32_t count_ready_target = (round_idx + 1u) * active_dispatch_warps;
    if (lane_idx == 0) {
      atomicAdd(const_cast<uint32_t *>(&sync_state[3]), 1u);
    }
    while (sync_state[3] < count_ready_target) {
    }
    __syncwarp();

    if (warp_idx == 0) {
      for (uint32_t expert = lane_idx;
           expert < static_cast<uint32_t>(num_experts) && expert < kNumExpertsLimit;
           expert += 32) {
        uint32_t slot = local_slots[expert];
        for (uint32_t w = 0; w < active_dispatch_warps; ++w) {
          const uint32_t idx = w * static_cast<uint32_t>(kNumExpertsLimit) + expert;
          round_slots[idx] = slot;
          slot += round_counts[idx];
        }
        local_slots[expert] = slot;
      }
      __syncwarp();
      if (lane_idx == 0) {
        sync_state[4] = round_idx + 1u;
      }
    } else {
      while (sync_state[4] < round_idx + 1u) {
      }
    }
    __syncwarp();

    if (expert_idx >= 0 && expert_idx < num_experts &&
        expert_idx < kNumExpertsLimit) {
      const uint32_t expert = static_cast<uint32_t>(expert_idx);
      const uint32_t dst_rank = expert / epr;
      const uint32_t dst_local_expert = expert - dst_rank * epr;
      const uint32_t dst_slot = atomicAdd(my_round_slots + expert, 1u);
      uint32_t *remote_queue = src_token_topk_idx_ptr(
          symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
          dst_local_expert, rank, dst_slot);
      nvshmem_uint_p(remote_queue, token_topk_idx, dst_rank);
    }
    __syncwarp();

    const uint32_t write_done_target = (round_idx + 1u) * active_dispatch_warps;
    if (lane_idx == 0) {
      atomicAdd(const_cast<uint32_t *>(&sync_state[5]), 1u);
    }
    while (sync_state[5] < write_done_target) {
    }
    __syncwarp();
  }
  nvshmem_quiet();

  if (warp_idx == 0) {
    for (uint32_t expert = lane_idx;
         expert < static_cast<uint32_t>(num_experts) && expert < kNumExpertsLimit;
         expert += 32) {
      const uint64_t expert_status = *expert_send_count_ptr(symm_buffer, expert);
      const uint32_t dst_rank = expert / epr;
      const uint32_t dst_local_expert = expert - dst_rank * epr;
      nvshmem_uint64_p(
          expert_recv_count_ptr(symm_buffer, num_experts, epr, rank,
                                dst_local_expert),
          expert_status & 0xffffffffu, dst_rank);
      nvshmem_quiet();
      nvshmem_uint64_atomic_add(
          expert_recv_count_sum_ptr(symm_buffer, num_experts, epr,
                                    dst_local_expert),
          expert_status, dst_rank);
    }
    __syncwarp();
    nvshmem_quiet();
  }
}

extern "C" __device__ void userhopper_ws_receiver_partition(
    uint8_t *symm_buffer,
    int expected_local_recv_tokens,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int num_padded_sf_pool_tokens) {
  if ((threadIdx.x & 31) != 0 || blockIdx.x != 0) {
    return;
  }

  const uint32_t rank = static_cast<uint32_t>(nvshmem_my_pe());
  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  uint8_t *l1_tokens =
      symm_buffer + l1_token_offset(num_ranks, num_experts,
                                    num_max_tokens_per_rank, num_topk, hidden);
  uint8_t *l1_sf =
      symm_buffer + l1_sf_offset(num_ranks, num_experts,
                                 num_max_tokens_per_rank, num_topk, hidden);
  float *l1_topk_weights = reinterpret_cast<float *>(
      symm_buffer + l1_topk_weight_offset(num_ranks, num_experts,
                                          num_max_tokens_per_rank, num_topk,
                                          hidden, num_padded_sf_pool_tokens));
  const uint64_t remote_x_off =
      input_token_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk);
  const uint64_t remote_sf_off =
      input_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden);
  const uint64_t remote_weight_off = input_topk_weight_offset(
      num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden);
  const uint32_t sf_count = hidden / 128;
  const uint32_t input_sf_bytes = hidden / 32;
  uint32_t pool_token_base = 0;
  volatile uint32_t *configured_expected =
      reinterpret_cast<volatile uint32_t *>(symm_buffer);
  const bool use_configured_expected = configured_expected[7] == 0x45585043u;

  for (uint32_t local_expert = 0; local_expert < epr; ++local_expert) {
    uint32_t expected_for_expert =
        use_configured_expected ? configured_expected[local_expert] : 0;
    if (!use_configured_expected) {
      for (uint32_t topk = 0; topk < static_cast<uint32_t>(num_topk); ++topk) {
        if ((topk % epr) == local_expert) {
          expected_for_expert +=
              static_cast<uint32_t>(expected_local_recv_tokens) /
              static_cast<uint32_t>(num_topk);
        }
      }
    }
    if (expected_for_expert == 0) {
      continue;
    }

    volatile uint64_t *sum_ptr = reinterpret_cast<volatile uint64_t *>(
        expert_recv_count_sum_ptr(symm_buffer, num_experts, epr,
                                  local_expert));
    uint64_t observed = 0;
    while (true) {
      observed = *sum_ptr;
      if ((observed & 0xffffffffu) >= expected_for_expert) {
        break;
      }
    }

    uint32_t rank_counts[8];
    uint32_t total = 0;
    while (true) {
      total = 0;
      for (uint32_t r = 0; r < static_cast<uint32_t>(num_ranks); ++r) {
        const uint64_t count = *reinterpret_cast<volatile uint64_t *>(
            expert_recv_count_ptr(symm_buffer, num_experts, epr, r,
                                  local_expert));
        rank_counts[r] = static_cast<uint32_t>(count & 0xffffffffu);
        total += rank_counts[r];
      }
      if (total >= expected_for_expert) {
        break;
      }
    }

    const uint32_t expert_block_offset = pool_token_base / kBlockM;
    for (uint32_t token_idx_in_expert = 0; token_idx_in_expert < total;
         ++token_idx_in_expert) {
      uint32_t src_rank = 0;
      uint32_t token_idx_in_rank = 0;
      choose_rank_round_robin(token_idx_in_expert, rank_counts, num_ranks,
                              &src_rank, &token_idx_in_rank);

      volatile uint32_t *queue_ptr = reinterpret_cast<volatile uint32_t *>(
          src_token_topk_idx_ptr(
              symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank,
              num_topk, local_expert, src_rank, token_idx_in_rank));
      const uint32_t src_token_topk_idx = *queue_ptr;
      const uint32_t src_token = src_token_topk_idx / num_topk;
      const uint32_t src_topk = src_token_topk_idx - src_token * num_topk;
      const uint32_t pool_idx = pool_token_base + token_idx_in_expert;

      nvshmem_getmem(l1_tokens + static_cast<uint64_t>(pool_idx) * hidden,
                     symm_buffer + remote_x_off +
                         static_cast<uint64_t>(src_token) * hidden,
                     static_cast<size_t>(hidden), src_rank);
      for (uint32_t sf_idx = 0; sf_idx < sf_count; ++sf_idx) {
        nvshmem_getmem(
            l1_sf + (static_cast<uint64_t>(sf_idx) * num_padded_sf_pool_tokens +
                     pool_idx) *
                        sizeof(float),
            symm_buffer + remote_sf_off +
                static_cast<uint64_t>(src_token) * input_sf_bytes +
                static_cast<uint64_t>(sf_idx) * sizeof(float),
            sizeof(float), src_rank);
      }

      float *remote_weight =
          reinterpret_cast<float *>(symm_buffer + remote_weight_off) +
          src_token_topk_idx;
      l1_topk_weights[pool_idx] = nvshmem_float_g(remote_weight, src_rank);

      uint32_t *metadata = token_src_metadata_ptr(
          symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank,
          num_topk, pool_idx);
      metadata[0] = src_rank;
      metadata[1] = src_token;
      metadata[2] = src_topk;
      atomicAdd(l1_arrival_count_ptr(
                    symm_buffer, num_experts, epr,
                    expert_block_offset + token_idx_in_expert / kBlockM),
                1u);
    }
    pool_token_base += align_u32(total, kBlockM);
  }
}

extern "C" __device__ void userhopper_ws_compute_stub_partition(
    uint8_t *symm_buffer,
    uint8_t *l1_weights,
    float *l1_weights_sf,
    uint8_t *l2_weights,
    float *l2_weights_sf,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    int num_padded_sf_pool_tokens,
    int compute_full_hidden,
    int compute_parallel,
    int compute_worker_warps) {
  const uint32_t lane_idx = static_cast<uint32_t>(threadIdx.x & 31);
  const uint32_t active_compute_warps =
      compute_worker_warps > 0 ? static_cast<uint32_t>(compute_worker_warps) : 1u;
  const uint32_t worker_warp_idx =
      static_cast<uint32_t>(threadIdx.x >> 5) % active_compute_warps;
  if (blockIdx.x != 0 ||
      (compute_parallel == 0 && (worker_warp_idx != 0 || lane_idx != 0))) {
    return;
  }

  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  volatile uint32_t *debug = reinterpret_cast<volatile uint32_t *>(symm_buffer);
  const bool use_configured_expected = debug[7] == 0x45585043u;
  if (!use_configured_expected) {
    if (worker_warp_idx == 0 && lane_idx == 0) {
      debug[4] = 0xffffffffu;
      debug[5] = 0xffffffffu;
      debug[6] = 0xBAD00001u;
    }
    return;
  }

  uint8_t *l1_tokens =
      symm_buffer + l1_token_offset(num_ranks, num_experts,
                                    num_max_tokens_per_rank, num_topk, hidden);
  float *l1_sf = reinterpret_cast<float *>(
      symm_buffer + l1_sf_offset(num_ranks, num_experts,
                                 num_max_tokens_per_rank, num_topk, hidden));
  float *l1_topk_weights = reinterpret_cast<float *>(
      symm_buffer + l1_topk_weight_offset(num_ranks, num_experts,
                                          num_max_tokens_per_rank, num_topk,
                                          hidden, num_padded_sf_pool_tokens));
  uint8_t *l2_tokens =
      symm_buffer + l2_token_offset(num_ranks, num_experts,
                                    num_max_tokens_per_rank, num_topk, hidden,
                                    num_padded_sf_pool_tokens);
  float *l2_sf = reinterpret_cast<float *>(
      symm_buffer +
      l2_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                   hidden, static_cast<uint32_t>(intermediate_hidden),
                   num_padded_sf_pool_tokens));
  const uint64_t l2_token_bytes = static_cast<uint64_t>(
      num_max_pool_tokens(static_cast<uint32_t>(num_ranks),
                          static_cast<uint32_t>(num_max_tokens_per_rank),
                          static_cast<uint32_t>(num_topk), epr)) *
      static_cast<uint32_t>(intermediate_hidden);
  uint32_t *l2_debug = reinterpret_cast<uint32_t *>(
      l2_tokens + l2_token_bytes - 8u * sizeof(uint32_t));
  const uint64_t combine_base_off = combine_token_offset(
      static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
      static_cast<uint32_t>(num_max_tokens_per_rank),
      static_cast<uint32_t>(num_topk), static_cast<uint32_t>(hidden),
      static_cast<uint32_t>(intermediate_hidden),
      static_cast<uint32_t>(num_padded_sf_pool_tokens));
  __shared__ float parallel_l1_values[64];
  __shared__ float parallel_l2_scale;
  __shared__ volatile uint32_t compute_sync[5];
  if (compute_parallel != 0) {
    if (worker_warp_idx == 0 && lane_idx < 5u) {
      compute_sync[lane_idx] = 0;
    }
    __syncwarp();
    if (worker_warp_idx == 0 && lane_idx == 0) {
      compute_sync[0] = 1;
    } else {
      while (compute_sync[0] == 0) {
      }
    }
    __syncwarp();
  }

  uint32_t expected_total = 0;
  uint32_t observed_total = 0;
  uint32_t checksum = 2166136261u;
  uint32_t l1_weight_checksum = 2166136261u;
  float l1_scalar_sum = 0.0f;
  uint32_t pool_token_base = 0;
  uint32_t compute_barrier_epoch = 0;
  uint32_t compute_token_epoch = 0;
  for (uint32_t local_expert = 0; local_expert < epr; ++local_expert) {
    const uint32_t expected_for_expert = debug[local_expert];
    if (worker_warp_idx == 0 && lane_idx == 0) {
      expected_total += expected_for_expert;
    }
    const uint32_t first_block = pool_token_base / kBlockM;
    const uint32_t num_blocks =
        expected_for_expert == 0 ? 0 : ((expected_for_expert + kBlockM - 1) / kBlockM);
    if (worker_warp_idx == 0 && lane_idx == 0) {
      for (uint32_t block = 0; block < num_blocks; ++block) {
        observed_total += *reinterpret_cast<volatile uint32_t *>(
            l1_arrival_count_ptr(symm_buffer, num_experts, epr, first_block + block));
      }
    }
    for (uint32_t token_idx_in_expert = 0; token_idx_in_expert < expected_for_expert;
         ++token_idx_in_expert) {
      const uint32_t pool_idx = pool_token_base + token_idx_in_expert;
      const uint32_t *metadata = token_src_metadata_ptr(
          symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
          pool_idx);
      const uint32_t src_rank_for_combine = metadata[0];
      const uint32_t src_token_for_combine = metadata[1];
      const uint32_t src_topk_for_combine = metadata[2];
      const float topk_weight_for_l2 = l1_topk_weights[pool_idx];
      const uint32_t compute_h =
          compute_full_hidden != 0 ? static_cast<uint32_t>(hidden)
                                   : (hidden < 32 ? hidden : 32);
      const uint32_t compute_i = static_cast<uint32_t>(intermediate_hidden);
      const uint64_t expert_weight_base =
          static_cast<uint64_t>(local_expert) *
          (2u * static_cast<uint32_t>(intermediate_hidden)) *
          static_cast<uint32_t>(hidden);
      const uint64_t gate_weight_base = expert_weight_base;
      const uint32_t sf_stride =
          (2u * static_cast<uint32_t>(intermediate_hidden) / 128u) *
          (static_cast<uint32_t>(hidden) / 128u);
      const uint64_t sf_base = static_cast<uint64_t>(local_expert) * sf_stride;
      if (worker_warp_idx == 0 && lane_idx == 0) {
        checksum ^= static_cast<uint32_t>(
            l1_tokens[static_cast<uint64_t>(pool_idx) * hidden]);
        checksum *= 16777619u;
        checksum ^= __float_as_uint(l1_topk_weights[pool_idx]);
        checksum *= 16777619u;
        uint32_t weighted = 0;
        for (uint32_t h = 0; h < compute_h; ++h) {
          const uint8_t token_raw =
              l1_tokens[static_cast<uint64_t>(pool_idx) * hidden + h];
          const uint32_t token_byte = static_cast<uint32_t>(token_raw);
          const uint32_t weight_byte = static_cast<uint32_t>(
              l1_weights[gate_weight_base + h]);
          weighted += token_byte * weight_byte;
        }
        const float sf = l1_weights_sf
            [static_cast<uint64_t>(local_expert) * sf_stride];
        l1_weight_checksum ^= weighted;
        l1_weight_checksum *= 16777619u;
        l1_weight_checksum ^= __float_as_uint(sf);
        l1_weight_checksum *= 16777619u;
      }
      const uint32_t num_l2_sf_groups = compute_i / 64u;
      for (uint32_t sf_group = 0; sf_group < num_l2_sf_groups; ++sf_group) {
        const uint32_t first_in_group =
            compute_parallel != 0 ? worker_warp_idx * 32u + lane_idx : 0u;
        const uint32_t in_group_stride =
            compute_parallel != 0 ? active_compute_warps * 32u : 1u;
        for (uint32_t in_group = first_in_group; in_group < 64u;
             in_group += in_group_stride) {
          const uint32_t ii = sf_group * 64u + in_group;
          const uint32_t group = ii / 8u;
          const uint32_t lane = ii - group * 8u;
          const uint32_t gate_row = group * 16u + lane;
          const uint32_t up_row = gate_row + 8u;
          const uint64_t gate_row_base =
              expert_weight_base + static_cast<uint64_t>(gate_row) * hidden;
          const uint64_t up_row_base =
              expert_weight_base + static_cast<uint64_t>(up_row) * hidden;
          const float gate_sf_for_i = l1_weights_sf
              [sf_base + (ii / 128u) * (static_cast<uint32_t>(hidden) / 128u)];
          const float up_sf_for_i = l1_weights_sf
              [sf_base +
               ((static_cast<uint32_t>(intermediate_hidden) + ii) / 128u) *
                   (static_cast<uint32_t>(hidden) / 128u)];
          float gate_acc = 0.0f;
          float up_acc = 0.0f;
          for (uint32_t h = 0; h < compute_h; ++h) {
            const uint8_t token_raw =
                l1_tokens[static_cast<uint64_t>(pool_idx) * hidden + h];
            const float token_sf_for_h =
                l1_sf[static_cast<uint64_t>(h / 128u) *
                          static_cast<uint32_t>(num_padded_sf_pool_tokens) +
                      pool_idx];
            const float token_value =
                fp8_e4m3fn_to_float(token_raw) * token_sf_for_h;
            const uint32_t h_sf_block = h / 128u;
            const float gate_sf_for_h = l1_weights_sf
                [sf_base +
                 (ii / 128u) * (static_cast<uint32_t>(hidden) / 128u) +
                 h_sf_block];
            const float up_sf_for_h = l1_weights_sf
                [sf_base +
                 ((static_cast<uint32_t>(intermediate_hidden) + ii) / 128u) *
                     (static_cast<uint32_t>(hidden) / 128u) +
                 h_sf_block];
            gate_acc +=
                token_value * fp8_e4m3fn_to_float(l1_weights[gate_row_base + h]) *
                (compute_full_hidden != 0 ? gate_sf_for_h : gate_sf_for_i);
            up_acc +=
                token_value * fp8_e4m3fn_to_float(l1_weights[up_row_base + h]) *
                (compute_full_hidden != 0 ? up_sf_for_h : up_sf_for_i);
          }
          const float sigmoid_gate = 1.0f / (1.0f + expf(-gate_acc));
          const float swiglu = gate_acc * sigmoid_gate * up_acc;
          const float weighted_swiglu = swiglu * topk_weight_for_l2;
          if (compute_parallel != 0) {
            parallel_l1_values[in_group] = weighted_swiglu;
          } else {
            const float abs_value = fabsf(weighted_swiglu);
            parallel_l1_values[in_group] = weighted_swiglu;
            parallel_l2_scale =
                in_group == 0 || abs_value > parallel_l2_scale ? abs_value : parallel_l2_scale;
          }
        }
        if (compute_parallel != 0) {
          __syncwarp();
          if (lane_idx == 0) {
            atomicAdd(const_cast<uint32_t *>(&compute_sync[1]), 1u);
          }
          const uint32_t target = (compute_barrier_epoch + 1u) * active_compute_warps;
          while (compute_sync[1] < target) {
          }
        }
        if (worker_warp_idx == 0 && lane_idx == 0) {
          float max_abs = 0.0f;
          for (uint32_t in_group = 0; in_group < 64u; ++in_group) {
            l1_scalar_sum += parallel_l1_values[in_group];
            const float abs_value = fabsf(parallel_l1_values[in_group]);
            max_abs = abs_value > max_abs ? abs_value : max_abs;
          }
          parallel_l2_scale = max_abs > 0.0f ? max_abs / 448.0f : 1.0f;
          l2_sf[static_cast<uint64_t>(sf_group) *
                    static_cast<uint32_t>(num_padded_sf_pool_tokens) +
                pool_idx] = parallel_l2_scale;
        }
        if (compute_parallel != 0) {
          __syncwarp();
          if (lane_idx == 0) {
            atomicAdd(const_cast<uint32_t *>(&compute_sync[2]), 1u);
          }
          const uint32_t target = (compute_barrier_epoch + 1u) * active_compute_warps;
          while (compute_sync[2] < target) {
          }
        }
        for (uint32_t in_group = first_in_group; in_group < 64u;
             in_group += in_group_stride) {
          const uint32_t ii = sf_group * 64u + in_group;
          l2_tokens[static_cast<uint64_t>(pool_idx) *
                        static_cast<uint32_t>(intermediate_hidden) +
                    ii] = __nv_cvt_float_to_fp8(
              parallel_l1_values[in_group] / parallel_l2_scale,
              __NV_SATFINITE, __NV_E4M3);
        }
        if (compute_parallel != 0) {
          __syncwarp();
          if (lane_idx == 0) {
            atomicAdd(const_cast<uint32_t *>(&compute_sync[3]), 1u);
          }
          const uint32_t target = (compute_barrier_epoch + 1u) * active_compute_warps;
          while (compute_sync[3] < target) {
          }
        }
        ++compute_barrier_epoch;
      }
      const uint64_t l2_weight_expert_base =
          static_cast<uint64_t>(local_expert) * static_cast<uint32_t>(hidden) *
          static_cast<uint32_t>(intermediate_hidden);
      const uint32_t l2_sf_stride =
          (static_cast<uint32_t>(hidden) / 128u) *
          (static_cast<uint32_t>(intermediate_hidden) / 128u);
      const uint64_t l2_sf_base = static_cast<uint64_t>(local_expert) * l2_sf_stride;
      if (compute_parallel != 0) {
        __syncwarp();
        if (lane_idx == 0) {
          atomicAdd(const_cast<uint32_t *>(&compute_sync[4]), 1u);
        }
        const uint32_t target = (compute_token_epoch + 1u) * active_compute_warps;
        while (compute_sync[4] < target) {
        }
        ++compute_token_epoch;
      }
      const uint32_t first_h_pair =
          compute_parallel != 0 ? (worker_warp_idx * 32u + lane_idx) * 2u : 0u;
      const uint32_t h_pair_stride =
          compute_parallel != 0 ? active_compute_warps * 64u : 2u;
      for (uint32_t h_pair = first_h_pair;
           h_pair < static_cast<uint32_t>(hidden); h_pair += h_pair_stride) {
        float accum0 = 0.0f;
        float accum1 = 0.0f;
        for (uint32_t ii = 0; ii < compute_i; ++ii) {
          const uint32_t sf_group = ii / 64u;
          const float act_scale =
              l2_sf[static_cast<uint64_t>(sf_group) *
                        static_cast<uint32_t>(num_padded_sf_pool_tokens) +
                    pool_idx];
          const float act_value =
              fp8_e4m3fn_to_float(l2_tokens[static_cast<uint64_t>(pool_idx) *
                                                static_cast<uint32_t>(intermediate_hidden) +
                                            ii]) *
              act_scale;
          const uint32_t sf_i_block = ii / 128u;
          const uint32_t sf_h_block0 = h_pair / 128u;
          const uint32_t sf_h_block1 = (h_pair + 1u) / 128u;
          const float weight_sf0 =
              l2_weights_sf[l2_sf_base +
                            sf_h_block0 *
                                (static_cast<uint32_t>(intermediate_hidden) / 128u) +
                            sf_i_block];
          const float weight_sf1 =
              l2_weights_sf[l2_sf_base +
                            sf_h_block1 *
                                (static_cast<uint32_t>(intermediate_hidden) / 128u) +
                            sf_i_block];
          accum0 += act_value *
                    fp8_e4m3fn_to_float(
                        l2_weights[l2_weight_expert_base +
                                   static_cast<uint64_t>(h_pair) *
                                       static_cast<uint32_t>(intermediate_hidden) +
                                   ii]) *
                    weight_sf0;
          accum1 += act_value *
                    fp8_e4m3fn_to_float(
                        l2_weights[l2_weight_expert_base +
                                   static_cast<uint64_t>(h_pair + 1u) *
                                       static_cast<uint32_t>(intermediate_hidden) +
                                   ii]) *
                    weight_sf1;
        }
        const uint32_t packed =
            static_cast<uint32_t>(float_to_bf16_bits(accum0)) |
            (static_cast<uint32_t>(float_to_bf16_bits(accum1)) << 16);
        uint32_t *remote_combine = reinterpret_cast<uint32_t *>(
            symm_buffer + combine_base_off +
            (static_cast<uint64_t>(src_topk_for_combine) *
                 static_cast<uint32_t>(num_max_tokens_per_rank) +
             src_token_for_combine) *
                static_cast<uint32_t>(hidden) * sizeof(uint16_t) +
            static_cast<uint64_t>(h_pair) * sizeof(uint16_t));
        nvshmem_uint_p(remote_combine, packed, static_cast<int>(src_rank_for_combine));
      }
      if (compute_parallel != 0) {
        __syncwarp();
      }
    }
    pool_token_base += align_u32(expected_for_expert, kBlockM);
  }

  nvshmem_quiet();
  if (worker_warp_idx == 0 && lane_idx == 0) {
    l2_debug[0] = checksum;
    l2_debug[1] = observed_total;
    l2_debug[2] = expected_total;
    l2_debug[3] = observed_total == expected_total ? 0xC0DEC0DEu : 0xBAD00002u;
    l2_debug[4] = l1_weight_checksum;
    l2_debug[5] = 0x1A10C001u;
    l2_debug[6] = __float_as_uint(l1_scalar_sum);
    l2_debug[7] = 0x51A10F32u;
    debug[4] = observed_total;
    debug[5] = expected_total;
    debug[6] = observed_total == expected_total ? 0xC0DEC0DEu : 0xBAD00002u;
  }
}

extern "C" __device__ void userhopper_ws_combine_reduce_partition(
    uint8_t *symm_buffer,
    uint8_t *y,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    int num_padded_sf_pool_tokens,
    int cleanup_workspace) {
  if ((threadIdx.x & 31) != 0 || blockIdx.x != 0) {
    return;
  }

  nvshmem_barrier_all();

  const uint64_t combine_base_off = combine_token_offset(
      static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
      static_cast<uint32_t>(num_max_tokens_per_rank),
      static_cast<uint32_t>(num_topk), static_cast<uint32_t>(hidden),
      static_cast<uint32_t>(intermediate_hidden),
      static_cast<uint32_t>(num_padded_sf_pool_tokens));
  uint16_t *combine_tokens = reinterpret_cast<uint16_t *>(
      symm_buffer + combine_base_off);
  uint16_t *y_bf16 = reinterpret_cast<uint16_t *>(y);
  for (uint32_t token = 0; token < static_cast<uint32_t>(num_max_tokens_per_rank);
       ++token) {
    for (uint32_t h_pair = 0; h_pair < static_cast<uint32_t>(hidden); h_pair += 2u) {
      float accum0 = 0.0f;
      float accum1 = 0.0f;
      for (uint32_t topk = 0; topk < static_cast<uint32_t>(num_topk); ++topk) {
        const uint64_t combine_idx =
            (static_cast<uint64_t>(topk) *
                 static_cast<uint32_t>(num_max_tokens_per_rank) +
             token) *
                static_cast<uint32_t>(hidden) +
            h_pair;
        accum0 += __bfloat162float(*reinterpret_cast<__nv_bfloat16 *>(
            combine_tokens + combine_idx));
        accum1 += __bfloat162float(*reinterpret_cast<__nv_bfloat16 *>(
            combine_tokens + combine_idx + 1u));
      }
      y_bf16[static_cast<uint64_t>(token) * static_cast<uint32_t>(hidden) +
             h_pair] = float_to_bf16_bits(accum0);
      y_bf16[static_cast<uint64_t>(token) * static_cast<uint32_t>(hidden) +
             h_pair + 1u] = float_to_bf16_bits(accum1);
    }
  }

  if (cleanup_workspace == 0) {
    return;
  }

  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  const uint32_t pool_blocks = num_max_pool_blocks(
      static_cast<uint32_t>(num_ranks),
      static_cast<uint32_t>(num_max_tokens_per_rank),
      static_cast<uint32_t>(num_topk),
      epr);

  for (uint32_t expert = 0; expert < static_cast<uint32_t>(num_experts); ++expert) {
    *expert_send_count_ptr(symm_buffer, expert) = 0;
  }
  for (uint32_t rank_idx = 0; rank_idx < static_cast<uint32_t>(num_ranks); ++rank_idx) {
    for (uint32_t local_expert = 0; local_expert < epr; ++local_expert) {
      *expert_recv_count_ptr(symm_buffer, static_cast<uint32_t>(num_experts), epr,
                             rank_idx, local_expert) = 0;
    }
  }
  for (uint32_t local_expert = 0; local_expert < epr; ++local_expert) {
    *expert_recv_count_sum_ptr(symm_buffer, static_cast<uint32_t>(num_experts), epr,
                               local_expert) = 0;
  }
  for (uint32_t block = 0; block < pool_blocks; ++block) {
    *l1_arrival_count_ptr(symm_buffer, static_cast<uint32_t>(num_experts), epr,
                          block) = 0;
    *l2_arrival_mask_ptr(symm_buffer, static_cast<uint32_t>(num_ranks),
                         static_cast<uint32_t>(num_experts),
                         static_cast<uint32_t>(num_max_tokens_per_rank),
                         static_cast<uint32_t>(num_topk), block) = 0;
  }
}

extern "C" __device__ void userhopper_ws_tldot_combine_write_partition(
    uint8_t *symm_buffer,
    float *l2_out,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    int num_padded_sf_pool_tokens) {
  if (blockIdx.x != 0) {
    return;
  }

  const uint32_t lane_idx = static_cast<uint32_t>(threadIdx.x & 31);
  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  volatile uint32_t *debug = reinterpret_cast<volatile uint32_t *>(symm_buffer);
  if (debug[7] != 0x45585043u) {
    return;
  }

  const uint64_t combine_base_off = combine_token_offset(
      static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
      static_cast<uint32_t>(num_max_tokens_per_rank),
      static_cast<uint32_t>(num_topk), static_cast<uint32_t>(hidden),
      static_cast<uint32_t>(intermediate_hidden),
      static_cast<uint32_t>(num_padded_sf_pool_tokens));

  uint32_t pool_token_base = 0;
  for (uint32_t local_expert = 0; local_expert < epr; ++local_expert) {
    const uint32_t expected_for_expert = debug[local_expert];
    for (uint32_t token_idx_in_expert = 0; token_idx_in_expert < expected_for_expert;
         ++token_idx_in_expert) {
      const uint32_t pool_idx = pool_token_base + token_idx_in_expert;
      const uint32_t *metadata = token_src_metadata_ptr(
          symm_buffer, static_cast<uint32_t>(num_ranks),
          static_cast<uint32_t>(num_experts),
          static_cast<uint32_t>(num_max_tokens_per_rank),
          static_cast<uint32_t>(num_topk), pool_idx);
      const uint32_t src_rank = metadata[0];
      const uint32_t src_token = metadata[1];
      const uint32_t src_topk = metadata[2];
      for (uint32_t h_pair = lane_idx * 2u; h_pair < static_cast<uint32_t>(hidden);
           h_pair += 64u) {
        const float value0 =
            l2_out[static_cast<uint64_t>(pool_idx) * static_cast<uint32_t>(hidden) +
                   h_pair];
        const float value1 =
            l2_out[static_cast<uint64_t>(pool_idx) * static_cast<uint32_t>(hidden) +
                   h_pair + 1u];
        const uint32_t packed =
            static_cast<uint32_t>(float_to_bf16_bits(value0)) |
            (static_cast<uint32_t>(float_to_bf16_bits(value1)) << 16);
        uint32_t *remote_combine = reinterpret_cast<uint32_t *>(
            symm_buffer + combine_base_off +
            (static_cast<uint64_t>(src_topk) *
                 static_cast<uint32_t>(num_max_tokens_per_rank) +
             src_token) *
                static_cast<uint32_t>(hidden) * sizeof(uint16_t) +
            static_cast<uint64_t>(h_pair) * sizeof(uint16_t));
        nvshmem_uint_p(remote_combine, packed, static_cast<int>(src_rank));
      }
    }
    pool_token_base += align_u32(expected_for_expert, kBlockM);
  }

  nvshmem_quiet();
}

extern "C" __device__ void userhopper_ws_receiver_partition_bounded(
    uint8_t *symm_buffer,
    int expected_local_recv_tokens,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int num_padded_sf_pool_tokens) {
  if ((threadIdx.x & 31) != 0 || blockIdx.x != 0) {
    return;
  }

  volatile uint32_t *debug = reinterpret_cast<volatile uint32_t *>(symm_buffer);
  debug[0] = 0xB001u;
  debug[6] = static_cast<uint32_t>(threadIdx.x);
  debug[7] = static_cast<uint32_t>(nvshmem_my_pe());

  const uint32_t rank = static_cast<uint32_t>(nvshmem_my_pe());
  const uint32_t epr = static_cast<uint32_t>(num_experts / num_ranks);
  const uint32_t local_expert = 0;
  volatile uint64_t *sum_ptr = reinterpret_cast<volatile uint64_t *>(
      expert_recv_count_sum_ptr(symm_buffer, num_experts, epr, local_expert));

  uint64_t observed = 0;
  uint32_t spins = 0;
  constexpr uint32_t kMaxSpins = 1u << 24;
  while (spins < kMaxSpins) {
    observed = *sum_ptr;
    debug[1] = static_cast<uint32_t>(observed & 0xffffffffu);
    debug[2] = static_cast<uint32_t>(observed >> 32);
    if ((observed & 0xffffffffu) >=
        static_cast<uint32_t>(expected_local_recv_tokens)) {
      break;
    }
    ++spins;
  }
  if (spins >= kMaxSpins) {
    debug[0] = 0xB0E1u;
    return;
  }

  uint32_t rank_counts[8];
  uint32_t total = 0;
  spins = 0;
  while (spins < kMaxSpins) {
    total = 0;
    for (uint32_t r = 0; r < static_cast<uint32_t>(num_ranks); ++r) {
      const uint64_t count = *reinterpret_cast<volatile uint64_t *>(
          expert_recv_count_ptr(symm_buffer, num_experts, epr, r,
                                local_expert));
      rank_counts[r] = static_cast<uint32_t>(count & 0xffffffffu);
      total += rank_counts[r];
      if (r < 2) {
        debug[3 + r] = rank_counts[r];
      }
    }
    debug[5] = total;
    if (total >= static_cast<uint32_t>(expected_local_recv_tokens)) {
      break;
    }
    ++spins;
  }
  if (spins >= kMaxSpins) {
    debug[0] = 0xB0E2u;
    return;
  }

  uint8_t *l1_tokens =
      symm_buffer + l1_token_offset(num_ranks, num_experts,
                                    num_max_tokens_per_rank, num_topk, hidden);
  uint8_t *l1_sf =
      symm_buffer + l1_sf_offset(num_ranks, num_experts,
                                 num_max_tokens_per_rank, num_topk, hidden);
  float *l1_topk_weights = reinterpret_cast<float *>(
      symm_buffer + l1_topk_weight_offset(num_ranks, num_experts,
                                          num_max_tokens_per_rank, num_topk,
                                          hidden, num_padded_sf_pool_tokens));
  const uint64_t remote_x_off =
      input_token_offset(num_ranks, num_experts, num_max_tokens_per_rank,
                         num_topk);
  const uint64_t remote_sf_off =
      input_sf_offset(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                      hidden);
  const uint64_t remote_weight_off = input_topk_weight_offset(
      num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden);
  const uint32_t sf_count = hidden / 128;
  const uint32_t input_sf_bytes = hidden / 32;

  for (uint32_t pool_idx = 0; pool_idx < total; ++pool_idx) {
    uint32_t src_rank = 0;
    uint32_t token_idx_in_rank = 0;
    choose_rank_round_robin(pool_idx, rank_counts, num_ranks, &src_rank,
                            &token_idx_in_rank);

    volatile uint32_t *queue_ptr = reinterpret_cast<volatile uint32_t *>(
        src_token_topk_idx_ptr(
        symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        local_expert, src_rank, token_idx_in_rank));
    const uint32_t src_token_topk_idx = *queue_ptr;
    const uint32_t src_token = src_token_topk_idx / num_topk;
    const uint32_t src_topk = src_token_topk_idx - src_token * num_topk;

    nvshmem_getmem(l1_tokens + static_cast<uint64_t>(pool_idx) * hidden,
                   symm_buffer + remote_x_off +
                       static_cast<uint64_t>(src_token) * hidden,
                   static_cast<size_t>(hidden), src_rank);
    for (uint32_t sf_idx = 0; sf_idx < sf_count; ++sf_idx) {
      nvshmem_getmem(
          l1_sf + (static_cast<uint64_t>(sf_idx) * num_padded_sf_pool_tokens +
                   pool_idx) *
                      sizeof(float),
          symm_buffer + remote_sf_off +
              static_cast<uint64_t>(src_token) * input_sf_bytes +
              static_cast<uint64_t>(sf_idx) * sizeof(float),
          sizeof(float), src_rank);
    }

    float *remote_weight =
        reinterpret_cast<float *>(symm_buffer + remote_weight_off) +
        src_token_topk_idx;
    l1_topk_weights[pool_idx] = nvshmem_float_g(remote_weight, src_rank);

    uint32_t *metadata = token_src_metadata_ptr(
        symm_buffer, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        pool_idx);
    metadata[0] = src_rank;
    metadata[1] = src_token;
    metadata[2] = src_topk;
    atomicAdd(l1_arrival_count_ptr(symm_buffer, num_experts, epr,
                                   pool_idx / kBlockM),
              1u);
  }
  debug[0] = 0xB0FFu;
}
