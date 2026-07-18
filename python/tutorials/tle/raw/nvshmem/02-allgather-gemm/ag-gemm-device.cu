#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __device__ void
nvshmemx_putmem_signal_nbi_block(void *dest, const void *source, size_t nbytes,
                                 uint64_t *sig_addr, uint64_t signal,
                                 int sig_op, int pe);

extern "C" __device__ uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr,
                                                         int cmp,
                                                         uint64_t cmp_value);

enum {
  NVSHMEM_CMP_GE = 5,
  NVSHMEM_SIGNAL_SET = 9,
};

extern "C" __device__ __attribute__((always_inline)) void
ag_mark_local_ready(__attribute__((address_space(1))) uint64_t *ready,
                    int rank) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    ready[(size_t)rank] = 1;
  }
  __syncthreads();
}

extern "C" __device__ __attribute__((always_inline)) void
ag_publish_local_rank(__attribute__((address_space(1))) __half *workspace,
                      __attribute__((address_space(1))) uint64_t *ready,
                      int elements_per_rank, int rank, int world_size) {
  int peer_offset = (int)blockIdx.x + 1;
  if (peer_offset >= world_size) {
    return;
  }

  int peer = (rank + peer_offset) % world_size;
  __half *local_rank = workspace + (size_t)rank * elements_per_rank;
  uint64_t *rank_ready = ready + (size_t)rank;

  nvshmemx_putmem_signal_nbi_block(local_rank, local_rank,
                                   (size_t)elements_per_rank * sizeof(__half),
                                   rank_ready, 1, NVSHMEM_SIGNAL_SET, peer);
}

extern "C" __device__ __attribute__((always_inline)) void
ag_wait_ready(__attribute__((address_space(1))) uint64_t *ready,
              int signal_index) {
  if (threadIdx.x == 0) {
    nvshmem_signal_wait_until(ready + signal_index, NVSHMEM_CMP_GE, 1);
  }
  __syncthreads();
}
