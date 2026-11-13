#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>
#include <stdint.h>

#define HIP_CHECK(stmt)                                                                            \
  do {                                                                                             \
    hipError_t result = (stmt);                                                                    \
    if (result != hipSuccess) {                                                                    \
      fprintf(stderr, "HIP %s\n", hipGetErrorString(result));                                      \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

extern "C" int gemm_ar_workspace_create(int elements, int element_size, void **workspace,
                                        uint64_t **ready, int num_tiles, int *mype, int *npes) {
  if (elements <= 0 || element_size <= 0 || num_tiles <= 0 || workspace == nullptr || ready == nullptr)
    return -1;
  *mype = dushmem_my_pe();
  *npes = dushmem_n_pes();
  HIP_CHECK(hipSetDevice(dushmem_team_my_pe(DUSHMEMX_TEAM_NODE)));

  *workspace = dushmem_malloc((size_t)elements * (size_t)element_size);
  *ready = (uint64_t *)dushmem_malloc((size_t)(*npes) * (size_t)num_tiles * sizeof(uint64_t));
  if (*workspace == nullptr || *ready == nullptr)
    return -2;
  HIP_CHECK(hipMemset(*ready, 0, (size_t)(*npes) * (size_t)num_tiles * sizeof(uint64_t)));
  dushmem_barrier_all();
  return 0;
}

extern "C" void *gemm_ar_peer_ptr(void *ptr, int peer) { return dushmem_ptr(ptr, peer); }

// dushmem_ptr is only non-null when the peer is P2P-mapped. That mapping exists
// for DUSHMEMX_TEAM_NODE, not across nodes (those puts go through the proxy).
extern "C" int gemm_ar_intra_node(void) {
  return dushmem_team_n_pes(DUSHMEMX_TEAM_NODE) == dushmem_n_pes();
}

__global__ void ar_mark_row_tiles(uint64_t *ready, int rank, int num_tiles, int tile_begin, int tile_count,
                                  uint64_t epoch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tile_count)
    return;
  uint64_t *slot = ready + static_cast<long long>(rank) * num_tiles + tile_begin + i;
  __hip_atomic_store(slot, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" int gemm_ar_mark_tiles(uint64_t *ready, int rank, int num_tiles, int tile_begin, int tile_count,
                                  uint64_t epoch, hipStream_t stream) {
  if (tile_count <= 0)
    return 0;
  constexpr int threads = 256;
  int blocks = (tile_count + threads - 1) / threads;
  ar_mark_row_tiles<<<blocks, threads, 0, stream>>>(ready, rank, num_tiles, tile_begin, tile_count, epoch);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "HIP %s\n", hipGetErrorString(err));
    return -1;
  }
  return 0;
}

extern "C" int gemm_ar_workspace_destroy(void *workspace, void *ready) {
  HIP_CHECK(hipDeviceSynchronize());
  dushmem_barrier_all();
  dushmem_free(workspace);
  dushmem_free(ready);
  return 0;
}
