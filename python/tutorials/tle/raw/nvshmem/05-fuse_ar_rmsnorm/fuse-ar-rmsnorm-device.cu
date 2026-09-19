

#include <cuda_bf16.h>
#include <cstring>
#include <stddef.h>
#include <stdint.h>

namespace {

constexpr uint32_t kHpcPoisonWord = 0x80000000u;
constexpr int kMaxWorldSize = 8;

union __align__(16) Pack128 {
  uint32_t data[4];
  uint16_t bf16[8];
};

static __device__ __attribute__((always_inline)) Pack128
load_pack128(const Pack128 *ptr) {
  Pack128 value;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.data[0]), "=r"(value.data[1]),
                 "=r"(value.data[2]), "=r"(value.data[3])
               : "l"(ptr)
               : "memory");
  return value;
}

static __device__ __attribute__((always_inline)) Pack128
load_pack128_volatile(const Pack128 *ptr) {
  Pack128 value;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.data[0]), "=r"(value.data[1]),
                 "=r"(value.data[2]), "=r"(value.data[3])
               : "l"(ptr)
               : "memory");
  return value;
}

static __device__ __attribute__((always_inline)) void
store_pack128_volatile(Pack128 *ptr, const Pack128 &value) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.data[0]), "r"(value.data[1]),
                 "r"(value.data[2]), "r"(value.data[3])
               : "memory");
}

static __device__ __attribute__((always_inline)) void
store_pack128_multicast(Pack128 *ptr, const Pack128 &value) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.data[0]), "r"(value.data[1]),
                 "r"(value.data[2]), "r"(value.data[3])
               : "memory");
}

static __device__ __attribute__((always_inline)) Pack128
hpc_poison_pack128() {
  Pack128 value;
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    value.data[word] = kHpcPoisonWord;
  }
  return value;
}

static __device__ __attribute__((always_inline)) void
normalize_bf16_negative_zero(Pack128 &value) {
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    if (value.bf16[element] == 0x8000u) {
      value.bf16[element] = 0;
    }
  }
}

union Bf16x2Bits {
  uint32_t bits;
  __nv_bfloat162_raw raw;
};

static __device__ __attribute__((always_inline)) Pack128
reduce_pack128_fp32_tpn(const Pack128 *values, int nranks) {
  float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};
  for (int source = 0; source < nranks; ++source) {
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      Bf16x2Bits bits;
      bits.bits = values[source].data[pair];
      float2 pair_value =
          __bfloat1622float2(__nv_bfloat162(bits.raw));
      accum[pair * 2] += pair_value.x;
      accum[pair * 2 + 1] += pair_value.y;
    }
  }

  Pack128 result;
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    Bf16x2Bits bits;
    bits.raw = static_cast<__nv_bfloat162_raw>(
        __floats2bfloat162_rn(accum[pair * 2], accum[pair * 2 + 1]));
    result.data[pair] = bits.bits;
  }
  return result;
}

}  // namespace

namespace {

__device__ __attribute__((always_inline)) void fused_ar_rmsnorm_allreduce_impl(
    const __nv_bfloat16 *input, const uint64_t *peer_scatter_ptrs,
    void *multicast_broadcast_void, int num_tokens, int packs_per_token,
    int rank, int world_size) {
  int token = (int)blockIdx.x;
  if (token >= num_tokens) {
    cudaTriggerProgrammaticLaunchCompletion();
    return;
  }

  int cluster_size = (int)gridDim.y;
  int block_rank = (int)blockIdx.y;
  int tid = (int)threadIdx.x;
  int nthreads = (int)blockDim.x;

  int chunk_packs = (packs_per_token + cluster_size - 1) / cluster_size;
  int pack_begin = block_rank * chunk_packs;
  int pack_end = min(pack_begin + chunk_packs, packs_per_token);

  Pack128 *multicast_broadcast =
      reinterpret_cast<Pack128 *>(multicast_broadcast_void);
  const Pack128 *input_packs = reinterpret_cast<const Pack128 *>(input);
  int owner = token % world_size;
  int local_token = token / world_size;

  cudaGridDependencySynchronize();

  // Scatter this CTA's hidden chunk.
  for (int pack = pack_begin + tid; pack < pack_end; pack += nthreads) {
    Pack128 contribution =
        load_pack128(input_packs + (size_t)token * packs_per_token + pack);
    normalize_bf16_negative_zero(contribution);
    Pack128 *scatter_destination = reinterpret_cast<Pack128 *>(
        (uintptr_t)peer_scatter_ptrs[owner]);
    size_t scatter_offset =
        ((size_t)local_token * world_size + rank) * packs_per_token + pack;
    store_pack128_volatile(scatter_destination + scatter_offset, contribution);
  }

  __syncthreads();

  if (owner == rank) {
    Pack128 *local_scatter = reinterpret_cast<Pack128 *>(
        (uintptr_t)peer_scatter_ptrs[rank]);
    for (int pack = pack_begin + tid; pack < pack_end; pack += nthreads) {
      Pack128 values[kMaxWorldSize];
      bool all_ready = false;
      while (!all_ready) {
        all_ready = true;
        for (int source = 0; source < world_size; ++source) {
          size_t source_offset =
              ((size_t)local_token * world_size + source) * packs_per_token +
              pack;
          values[source] =
              load_pack128_volatile(local_scatter + source_offset);
          all_ready &= values[source].data[0] != kHpcPoisonWord;
          all_ready &= values[source].data[1] != kHpcPoisonWord;
          all_ready &= values[source].data[2] != kHpcPoisonWord;
          all_ready &= values[source].data[3] != kHpcPoisonWord;
        }
      }

      Pack128 poison = hpc_poison_pack128();
      for (int source = 0; source < world_size; ++source) {
        size_t source_offset =
            ((size_t)local_token * world_size + source) * packs_per_token +
            pack;
        store_pack128_volatile(local_scatter + source_offset, poison);
      }

      Pack128 result = reduce_pack128_fp32_tpn(values, world_size);
      size_t broadcast_offset = (size_t)token * packs_per_token + pack;
      store_pack128_multicast(multicast_broadcast + broadcast_offset, result);
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace

extern "C" __device__ __attribute__((always_inline)) void
fused_ar_rmsnorm_allreduce_bf16(
    __attribute__((address_space(1))) const __nv_bfloat16 *input,
    __attribute__((address_space(1))) const uint64_t *peer_scatter_ptrs,
    __attribute__((address_space(1))) void *multicast_broadcast_void,
    int num_tokens, int packs_per_token, int rank, int world_size) {
  fused_ar_rmsnorm_allreduce_impl(
      input, peer_scatter_ptrs, multicast_broadcast_void, num_tokens,
      packs_per_token, rank, world_size);
}
