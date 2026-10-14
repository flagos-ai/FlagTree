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

__global__ void ag_store_release(uint64_t *slot, uint64_t value) {
  __hip_atomic_store(slot, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" int ag_gemm_workspace_create(int elements_per_rank, int element_size, int ready_slots,
                                        void **workspace, uint64_t **ready, int *mype, int *npes) {
  if (elements_per_rank <= 0 || element_size <= 0 || ready_slots <= 0 || workspace == nullptr ||
      ready == nullptr)
    return -1;
  *mype = dushmem_my_pe();
  *npes = dushmem_n_pes();
  HIP_CHECK(hipSetDevice(dushmem_team_my_pe(DUSHMEMX_TEAM_NODE)));

  size_t workspace_bytes = (size_t)(*npes) * (size_t)elements_per_rank * (size_t)element_size;
  *workspace = dushmem_malloc(workspace_bytes);
  *ready = (uint64_t *)dushmem_malloc((size_t)ready_slots * sizeof(uint64_t));
  if (*workspace == nullptr || *ready == nullptr)
    return -2;
  HIP_CHECK(hipMemset(*ready, 0, (size_t)ready_slots * sizeof(uint64_t)));
  dushmem_barrier_all();
  return 0;
}

extern "C" void *ag_gemm_peer_ptr(void *ptr, int peer) { return dushmem_ptr(ptr, peer); }

// dushmem_ptr is only non-null when the peer is P2P-mapped. That mapping exists
// for DUSHMEMX_TEAM_NODE, not across nodes (those puts go through the proxy).
extern "C" int ag_gemm_intra_node(void) {
  return dushmem_team_n_pes(DUSHMEMX_TEAM_NODE) == dushmem_n_pes();
}

extern "C" int ag_gemm_copy_on_stream(hipStream_t stream, void *dst, const void *src, size_t bytes) {
  HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
  return 0;
}

extern "C" int ag_gemm_signal_on_stream(hipStream_t stream, uint64_t *slot, uint64_t value) {
  ag_store_release<<<1, 1, 0, stream>>>(slot, value);
  HIP_CHECK(hipGetLastError());
  return 0;
}

// Host putmem goes through the proxy thread. Intra-node bulk copies use
// dushmem_ptr + hipMemcpyAsync (copy engine), then ag_gemm_signal_on_stream.
extern "C" int ag_gemm_signal_wait_on_stream(uint64_t *sig_addr, int cmp, uint64_t value,
                                             hipStream_t stream) {
  dushmemx_signal_wait_until_on_stream(sig_addr, cmp, value, stream);
  return 0;
}

extern "C" void ag_gemm_barrier_on_stream(hipStream_t stream) { dushmemx_barrier_all_on_stream(stream); }

extern "C" int ag_gemm_workspace_destroy(void *workspace, void *ready) {
  HIP_CHECK(hipDeviceSynchronize());
  dushmem_barrier_all();
  dushmem_free(workspace);
  dushmem_free(ready);
  return 0;
}
