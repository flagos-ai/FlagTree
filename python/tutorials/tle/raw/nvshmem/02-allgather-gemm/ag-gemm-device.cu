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

extern "C" __device__ void ag_mark_local_ready(uint64_t *ready, int rank,
                                               int num_chunks) {
  int chunk_id = (int)blockIdx.x;
  if (chunk_id >= num_chunks) {
    return;
  }
  if (threadIdx.x == 0) {
    __threadfence_system();
    ready[(size_t)rank * num_chunks + chunk_id] = 1;
  }
  __syncthreads();
}

// One Triton program publishes one chunk of this rank's A slice to one peer.
extern "C" __device__ void
ag_publish_local_chunk(__half *workspace, uint64_t *ready,
                       int elements_per_rank, int elements_per_chunk,
                       int num_chunks, int rank, int world_size) {
  int block_id = (int)blockIdx.x;
  int peer_offset = block_id / num_chunks + 1;
  int chunk_id = block_id % num_chunks;
  if (peer_offset >= world_size) {
    return;
  }

  int peer = (rank + peer_offset) % world_size;
  __half *local_chunk = workspace + (size_t)rank * elements_per_rank +
                        (size_t)chunk_id * elements_per_chunk;
  uint64_t *chunk_ready = ready + (size_t)rank * num_chunks + chunk_id;

  nvshmemx_putmem_signal_nbi_block(local_chunk, local_chunk,
                                   (size_t)elements_per_chunk * sizeof(__half),
                                   chunk_ready, 1, NVSHMEM_SIGNAL_SET, peer);
}

// The GEMM program waits for the source chunk whose A rows it will consume.
extern "C" __device__ void ag_wait_ready(uint64_t *ready, int signal_index) {
  if (threadIdx.x == 0) {
    nvshmem_signal_wait_until(ready + signal_index, NVSHMEM_CMP_GE, 1);
  }
  __syncthreads();
}
