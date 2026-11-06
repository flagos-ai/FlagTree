#include <hip/hip_runtime.h>
#include <stdint.h>

// Per-rank ready flags. Triton waits here before hipBLAS consumes that rank's
// rows, so local GEMM can run while the copy engine is still pulling remotes.

extern "C" __device__ __attribute__((used, always_inline)) void
ag_mark_local_ready(uint64_t *ready, int rank, uint64_t epoch) {
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_store(ready + rank, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" __device__ __attribute__((used, always_inline)) void
ag_wait_ready(uint64_t *ready, int signal_index, uint64_t epoch) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(ready + signal_index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
    }
  }
  __syncthreads();
}
