#include <stddef.h>
#include <stdint.h>

extern "C" __device__ uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr,
                                                         int cmp,
                                                         uint64_t cmp_value);
extern "C" __device__ void
nvshmemx_putmem_signal_block(void *dest, const void *source, size_t bytes,
                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                             int pe);

enum {
  NVSHMEM_CMP_GE = 5,
  NVSHMEM_SIGNAL_SET = 9,
};

extern "C" __device__ __attribute__((always_inline)) void
ag_mark_local_ready(__attribute__((address_space(1))) uint64_t *ready,
                    int rank) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    ready[rank] = 1;
  }
  __syncthreads();
}

extern "C" __device__ __attribute__((always_inline)) void
ag_wait_ready(__attribute__((address_space(1))) uint64_t *ready,
              int signal_index) {
  if (threadIdx.x == 0) {
    nvshmem_signal_wait_until(ready + signal_index, NVSHMEM_CMP_GE, 1);
  }
  __syncthreads();
}

extern "C" __device__ __attribute__((always_inline)) void
ag_putmem_signal_block(__attribute__((address_space(1))) void *dest,
                       __attribute__((address_space(1))) const void *source,
                       size_t bytes,
                       __attribute__((address_space(1))) uint64_t *sig_addr,
                       uint64_t signal, int peer) {
  nvshmemx_putmem_signal_block(dest, source, bytes, sig_addr, signal,
                               NVSHMEM_SIGNAL_SET, peer);
}
