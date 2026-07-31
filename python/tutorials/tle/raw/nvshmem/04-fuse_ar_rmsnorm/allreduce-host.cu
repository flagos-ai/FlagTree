#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

namespace {

constexpr size_t kPackBytes = 16;

#define CUDA_CHECK_RETURN(stmt)                                                \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(result));                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

} // namespace

// Layout of each PE's local allocation:
//
//   scratch[source_rank][pack_index]
//
// Every element is one aligned 16-byte pack.  nvshmem_malloc is collective, so
// every rank must call this function with exactly the same arguments.
extern "C" int allreduce_workspace_create(int world_size, int packs_per_rank,
                                          void **scratch) {
  if (world_size <= 0 || packs_per_rank <= 0 || scratch == nullptr) {
    return -1;
  }
  if (nvshmem_n_pes() != world_size) {
    return -2;
  }
  if (nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE) != world_size) {
    // The first version supports one NVLink/NVSwitch node only.
    return -3;
  }

  size_t total_packs = (size_t)world_size * (size_t)packs_per_rank;
  *scratch = nvshmem_malloc(total_packs * kPackBytes);
  if (*scratch == nullptr) {
    return -4;
  }

  // The allocation must exist on every PE before any rank asks for peer
  // pointers or launches a remote store.
  nvshmem_barrier_all();
  return 0;
}

extern "C" void *allreduce_peer_scratch_ptr(void *scratch, int peer) {
  if (scratch == nullptr || peer < 0 || peer >= nvshmem_n_pes()) {
    return nullptr;
  }
  return nvshmem_ptr(scratch, peer);
}

extern "C" int allreduce_workspace_destroy(void *scratch) {
  if (scratch == nullptr) {
    return -1;
  }

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  nvshmem_barrier_all();
  nvshmem_free(scratch);
  return 0;
}

extern "C" int allreduce_node_team_size() {
  return nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
}
