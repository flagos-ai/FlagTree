#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __device__ __attribute__((always_inline)) void
ar_multimem_ar_vector_multicast_store(__attribute__((address_space(1)))
                                      __half *mc_ptr,
                                      __attribute__((address_space(1)))
                                      __half *mc_out_ptr,
                                      int offset) {
  uint32_t v0, v1, v2, v3;
  asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.v4.f16x2 "
               "{%0, %1, %2, %3}, [%4];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
               : "l"(mc_ptr + offset)
               : "memory");
  asm volatile("multimem.st.global.v4.f16x2 [%0], {%1, %2, %3, %4};" ::"l"(
                   mc_out_ptr + offset),
               "r"(v0), "r"(v1), "r"(v2), "r"(v3)
               : "memory");
}

static __device__ __attribute__((always_inline)) void
ar_store_release_sys_u64(__attribute__((address_space(1))) uint64_t *ptr,
                         uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(value)
               : "memory");
}

static __device__ __attribute__((always_inline)) uint64_t
ar_load_acquire_sys_u64(__attribute__((address_space(1))) const uint64_t *ptr) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

extern "C" __device__ __attribute__((always_inline)) void
ar_mark_tile_ready(__attribute__((address_space(1))) uint64_t *ready,
                   int tile_id, int rank, int num_tiles, uint64_t epoch) {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    int offset = rank * num_tiles + tile_id;
    ar_store_release_sys_u64(ready + offset, epoch);
  }
}

extern "C" __device__ __attribute__((always_inline)) void
ar_wait_tile_ready(__attribute__((address_space(1)))
                   const uint64_t *peer_ready_ptrs,
                   int world_size, int tile_id, int num_tiles, uint64_t epoch) {
  if (threadIdx.x < world_size) {
    uintptr_t peer_base_addr = (uintptr_t)peer_ready_ptrs[threadIdx.x];
    __attribute__((address_space(1))) const uint64_t *peer_ready =
        (__attribute__((address_space(1))) const uint64_t *)peer_base_addr;
    __attribute__((address_space(1))) const uint64_t *slot =
        peer_ready + threadIdx.x * num_tiles + tile_id;
    while (ar_load_acquire_sys_u64(slot) < epoch) {
    }
  }
  __syncthreads();
}

extern "C" __device__ __attribute__((always_inline)) void
ar_multimem_store_barrier(__attribute__((address_space(1))) uint64_t *barrier,
                          __attribute__((address_space(1)))
                          const uint64_t *peer_barrier_ptrs,
                          int rank, int world_size, uint64_t epoch) {
  __syncthreads();

  if (threadIdx.x < world_size) {
    int num_comm_sms = gridDim.x;
    int comm_pid = blockIdx.x;
    uintptr_t peer_base_addr = (uintptr_t)peer_barrier_ptrs[threadIdx.x];
    __attribute__((address_space(1))) uint64_t *peer_barrier =
        (__attribute__((address_space(1))) uint64_t *)peer_base_addr;
    int remote_offset = rank * num_comm_sms + comm_pid;
    ar_store_release_sys_u64(peer_barrier + remote_offset, epoch);

    int local_offset = threadIdx.x * num_comm_sms + comm_pid;
    while (ar_load_acquire_sys_u64(barrier + local_offset) < epoch) {
    }
  }
}
