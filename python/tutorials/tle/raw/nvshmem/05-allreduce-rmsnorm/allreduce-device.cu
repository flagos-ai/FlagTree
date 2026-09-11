// Copyright 2026, The FlagOS Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kVectorWidth = 8;
constexpr int kMaxRanks = 8;

struct Float8 {
  float data[8];
};

union PackedBF16x2 {
  uint32_t bits;
  __nv_bfloat162 values;
};

__device__ __forceinline__ void fence_proxy_alias() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.alias;" ::: "memory");
#endif
}

__device__ __forceinline__ Float8 load_bf16x8(const __nv_bfloat16 *address) {
  PackedBF16x2 p0, p1, p2, p3;
  p0.values = *reinterpret_cast<const __nv_bfloat162 *>(address + 0);
  p1.values = *reinterpret_cast<const __nv_bfloat162 *>(address + 2);
  p2.values = *reinterpret_cast<const __nv_bfloat162 *>(address + 4);
  p3.values = *reinterpret_cast<const __nv_bfloat162 *>(address + 6);

  const float2 f0 = __bfloat1622float2(p0.values);
  const float2 f1 = __bfloat1622float2(p1.values);
  const float2 f2 = __bfloat1622float2(p2.values);
  const float2 f3 = __bfloat1622float2(p3.values);

  return {f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16 *address,
                                             const Float8 &value) {
  address[0] = __float2bfloat16_rn(value.data[0]);
  address[1] = __float2bfloat16_rn(value.data[1]);
  address[2] = __float2bfloat16_rn(value.data[2]);
  address[3] = __float2bfloat16_rn(value.data[3]);
  address[4] = __float2bfloat16_rn(value.data[4]);
  address[5] = __float2bfloat16_rn(value.data[5]);
  address[6] = __float2bfloat16_rn(value.data[6]);
  address[7] = __float2bfloat16_rn(value.data[7]);
}

__device__ __forceinline__ void add_inplace(Float8 &lhs, const Float8 &rhs) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lhs.data[i] += rhs.data[i];
  }
}

} // namespace

extern "C" __device__ __attribute__((always_inline)) void
fused_allreduce_bf16(__attribute__((address_space(1))) __nv_bfloat16 *reduced_out,
                     __attribute__((address_space(1))) const int64_t *peer_input_ptrs,
                     int rank,
                     int world_size,
                     int tokens) {
  if (reduced_out == nullptr || peer_input_ptrs == nullptr ||
      world_size <= 0 || world_size > kMaxRanks || tokens <= 0) {
    return;
  }
  (void)rank;

  const __nv_bfloat16 *inputs[kMaxRanks];
#pragma unroll
  for (int peer = 0; peer < kMaxRanks; ++peer) {
    if (peer < world_size) {
      inputs[peer] =
          reinterpret_cast<const __nv_bfloat16 *>(peer_input_ptrs[peer]);
    }
  }

  const int packed_per_row = kHiddenSize / kVectorWidth;

  fence_proxy_alias();
  __syncthreads();

  for (int row = 0; row < tokens; ++row) {
    const int base_row = row * kHiddenSize;
    for (int packed = threadIdx.x; packed < packed_per_row;
         packed += blockDim.x) {
      const int base = base_row + packed * kVectorWidth;
      Float8 reduced = load_bf16x8(inputs[0] + base);
#pragma unroll
      for (int peer = 1; peer < kMaxRanks; ++peer) {
        if (peer < world_size) {
          add_inplace(reduced, load_bf16x8(inputs[peer] + base));
        }
      }
      store_bf16x8(reduced_out + base, reduced);
    }
  }

  __syncthreads();
  fence_proxy_alias();
}
