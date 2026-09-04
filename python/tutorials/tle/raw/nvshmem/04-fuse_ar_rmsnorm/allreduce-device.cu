#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr uint32_t kPoisonWord = 0x80000000u;

// One pack is the 16-byte visibility unit used by the sentinel protocol.
union __align__(16) Pack128 {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
  } words;
  uint16_t bf16[8];
};

static __device__ __attribute__((always_inline)) Pack128
load_pack(const Pack128 *ptr) {
  Pack128 value;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.words.x), "=r"(value.words.y), "=r"(value.words.z),
                 "=r"(value.words.w)
               : "l"(ptr)
               : "memory");
  return value;
}

static __device__ __attribute__((always_inline)) Pack128
load_pack_volatile(const Pack128 *ptr) {
  Pack128 value;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.words.x), "=r"(value.words.y), "=r"(value.words.z),
                 "=r"(value.words.w)
               : "l"(ptr)
               : "memory");
  return value;
}

static __device__ __attribute__((always_inline)) void
store_pack_volatile(Pack128 *ptr, const Pack128 &value) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.words.x), "r"(value.words.y),
                 "r"(value.words.z), "r"(value.words.w)
               : "memory");
}

static __device__ __attribute__((always_inline)) void
normalize_negative_zero(Pack128 &value) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (value.bf16[i] == 0x8000u) {
      value.bf16[i] = 0;
    }
  }
}

} // namespace

// 1. Every rank writes its input pack into the rank-specific slot of every
//    peer's symmetric scratch allocation.
// 2. Every rank polls its local scratch until all source slots are valid.
// 3. Contributions are accumulated in FP32 and written as BF16.
template <int WorldSize>
static __device__ __attribute__((always_inline)) void
allreduce_one_shot_push_reduce_impl(
    __attribute__((address_space(1))) const __nv_bfloat16 *input,
    __attribute__((address_space(1))) __nv_bfloat16 *output,
    __attribute__((address_space(1))) const uint64_t *peer_scratch_ptrs,
    __attribute__((address_space(1))) Pack128 *local_scratch,
    int packs_per_rank, int rank) {
  static_assert(WorldSize > 0, "WorldSize must be positive");

  int global_thread = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total_threads = (int)(gridDim.x * blockDim.x);

  const Pack128 *input_packs = reinterpret_cast<const Pack128 *>(input);
  Pack128 *output_packs = reinterpret_cast<Pack128 *>(output);

  for (int pack_index = global_thread; pack_index < packs_per_rank;
       pack_index += total_threads) {
    Pack128 contribution = load_pack(input_packs + pack_index);
    normalize_negative_zero(contribution);

    // Push this rank's contribution to the same source-rank slot on every PE.
#pragma unroll
    for (int peer = 0; peer < WorldSize; ++peer) {
      uintptr_t peer_base = (uintptr_t)peer_scratch_ptrs[peer];
      __attribute__((address_space(1))) Pack128 *peer_scratch =
          (__attribute__((address_space(1))) Pack128 *)peer_base;
      int destination = rank * packs_per_rank + pack_index;
      store_pack_volatile(peer_scratch + destination, contribution);
    }

    float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // A 16-byte remote store is the publication event.  Once the first word
    // differs from the poison word, the complete pack is ready to consume.
#pragma unroll
    for (int source = 0; source < WorldSize; ++source) {
      int source_offset = source * packs_per_rank + pack_index;
      Pack128 value;
      do {
        value = load_pack_volatile(local_scratch + source_offset);
      } while (value.words.x == kPoisonWord);

#pragma unroll
      for (int element = 0; element < 8; ++element) {
        sum[element] +=
            __bfloat162float(__ushort_as_bfloat16(value.bf16[element]));
      }
    }

    Pack128 result;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      result.bf16[element] =
          __bfloat16_as_ushort(__float2bfloat16_rn(sum[element]));
    }
    store_pack_volatile(output_packs + pack_index, result);
  }
}

extern "C" __device__ __attribute__((always_inline)) void
allreduce_one_shot_push_reduce_tp8(
    __attribute__((address_space(1))) const __nv_bfloat16 *input,
    __attribute__((address_space(1))) __nv_bfloat16 *output,
    __attribute__((address_space(1))) const uint64_t *peer_scratch_ptrs,
    __attribute__((address_space(1))) Pack128 *local_scratch,
    int packs_per_rank, int rank) {
  allreduce_one_shot_push_reduce_impl<8>(input, output, peer_scratch_ptrs,
                                         local_scratch, packs_per_rank, rank);
}
