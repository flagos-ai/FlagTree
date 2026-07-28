#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(result));                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

extern "C" int gemm_ar_workspace_create(int elements_per_rank, void **workspace,
                                        uint64_t **ready, int num_tiles) {
    if (elements_per_rank <= 0 || workspace == nullptr || ready == nullptr) {
        return -1;
    }

    int npes = nvshmem_n_pes();
    size_t workspace_bytes = elements_per_rank * sizeof(__half);
    *workspace = nvshmem_malloc(workspace_bytes);
    *ready = (uint64_t *)nvshmem_calloc((size_t)npes * num_tiles, sizeof(uint64_t));

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

extern "C" int gemm_ar_multimem_output_create(int elements_per_rank, void **output,
                                               uint64_t **barrier,
                                               int num_comm_sms) {
    if (elements_per_rank <= 0 || output == nullptr || barrier == nullptr ||
        num_comm_sms <= 0) {
        return -1;
    }

    int npes = nvshmem_n_pes();
    size_t output_bytes = elements_per_rank * sizeof(__half);
    *output = nvshmem_malloc(output_bytes);
    *barrier = (uint64_t *)nvshmem_calloc(
        (size_t)npes * num_comm_sms, sizeof(uint64_t));

    if (*output == nullptr || *barrier == nullptr) {
        if (*barrier != nullptr) {
            nvshmem_free(*barrier);
        }
        if (*output != nullptr) {
            nvshmem_free(*output);
        }
        return -2;
    }

    nvshmem_barrier_all();
    return 0;
}

extern "C" void *gemm_ar_peer_workspace_ptr(void *workspace, int peer) {
    return nvshmem_ptr(workspace, peer);
}

extern "C" int gemm_ar_workspace_destroy(void *workspace, void *ready) {
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    nvshmem_free(workspace);
    nvshmem_free(ready);
    return 0;
}

extern "C" int gemm_ar_multimem_output_destroy(void *output, void *barrier) {
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    nvshmem_free(output);
    nvshmem_free(barrier);
    return 0;
}

extern "C" void *gemm_ar_mc_ptr(const void *workspace) {
    return nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, workspace);
}

extern "C" int gemm_ar_node_team_size() {
    return nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
}
