#include <hip/hip_runtime.h>
#include <stdint.h>

// Tile-ready flags only. The GEMM and the sum live in the Triton kernels.
// A system-scope release store lets the other rank's consumer observe the tile.

extern "C" __device__ __attribute__((used, always_inline)) void
ar_mark_tile_ready(uint64_t *ready, int tile_id, int rank, int num_tiles, uint64_t epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t *slot = ready + static_cast<long long>(rank) * num_tiles + tile_id;
    __hip_atomic_store(slot, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

extern "C" __device__ __attribute__((used, always_inline)) void
ar_wait_tile_ready(const uint64_t *peer_ready_ptrs, int world_size, int tile_id, int num_tiles,
                   uint64_t epoch) {
  if (threadIdx.x < world_size) {
    const uint64_t *peer_ready =
        reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(peer_ready_ptrs[threadIdx.x]));
    const uint64_t *slot = peer_ready + static_cast<long long>(threadIdx.x) * num_tiles + tile_id;
    while (__hip_atomic_load(const_cast<uint64_t *>(slot), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
           epoch) {
    }
  }
  __syncthreads();
}
