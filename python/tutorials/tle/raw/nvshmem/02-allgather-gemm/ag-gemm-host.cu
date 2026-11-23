#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(result));                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

extern "C" int ag_gemm_workspace_create(int elements_per_rank, int element_size,
                                        void **workspace, uint64_t **ready,
                                        int *mype, int *npes, int *mype_in_node,
                                        int *npes_in_node) {
  if (elements_per_rank <= 0 || element_size <= 0 || workspace == nullptr ||
      ready == nullptr) {
    return -1;
  }

  *mype = nvshmem_my_pe();
  *npes = nvshmem_n_pes();
  *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  *npes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  CUDA_CHECK(cudaSetDevice(*mype_in_node));

  size_t workspace_bytes = (size_t)(*npes) * elements_per_rank * element_size;
  *workspace = nvshmem_malloc(workspace_bytes);
  *ready = (uint64_t *)nvshmem_calloc((size_t)(*npes), sizeof(uint64_t));
  if (*workspace == nullptr || *ready == nullptr) {
    if (*ready != nullptr) {
      nvshmem_free(*ready);
    }
    if (*workspace != nullptr) {
      nvshmem_free(*workspace);
    }
    return -2;
  }

  nvshmem_barrier_all();
  return 0;
}

extern "C" int ag_gemm_workspace_destroy(void *workspace, void *ready) {
  CUDA_CHECK(cudaDeviceSynchronize());
  nvshmem_barrier_all();
  nvshmem_free(ready);
  nvshmem_free(workspace);
  return 0;
}

extern "C" void *ag_gemm_peer_workspace_ptr(void *workspace, int peer) {
  return nvshmem_ptr(workspace, peer);
}

extern "C" uint64_t *ag_gemm_peer_ready_ptr(uint64_t *ready, int peer) {
  return (uint64_t *)nvshmem_ptr(ready, peer);
}

extern "C" void ag_gemm_barrier_all_on_stream(cudaStream_t stream) {
  nvshmemx_barrier_all_on_stream(stream);
}

extern "C" void ag_gemm_signal_wait_until_on_stream(uint64_t *signal,
                                                    uint64_t value,
                                                    cudaStream_t stream) {
  nvshmemx_signal_wait_until_on_stream(signal, NVSHMEM_CMP_GE, value, stream);
}
