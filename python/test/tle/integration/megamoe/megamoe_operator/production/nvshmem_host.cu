#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stddef.h>

extern "C" void nvshmem_init_wrapper() { nvshmem_init(); }

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
  return nvshmemx_cumodule_init(module);
}

extern "C" int nvshmem_team_mype_wrapper() {
  return nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
}

extern "C" int nvshmem_mype_wrapper() { return nvshmem_my_pe(); }

extern "C" int nvshmem_npes_wrapper() { return nvshmem_n_pes(); }

extern "C" int *nvshmem_alloc_int_wrapper(int size) {
  return (int *)nvshmem_malloc(sizeof(int) * size);
}

extern "C" void *nvshmem_alloc_bytes_wrapper(size_t size) {
  return nvshmem_malloc(size);
}

extern "C" void *nvshmem_ptr_wrapper(const void *dest, int pe) {
  return nvshmem_ptr(dest, pe);
}

extern "C" void nvshmemx_barrier_wrapper(cudaStream_t stream) {
  nvshmemx_barrier_all_on_stream(stream);
}

extern "C" void nvshmem_free_int_wrapper(int *dest) { nvshmem_free(dest); }

extern "C" void nvshmem_free_ptr_wrapper(void *dest) { nvshmem_free(dest); }

extern "C" void nvshmem_finalize_wrapper() { nvshmem_finalize(); }

extern "C" void nvshmem_finalize_int_wrapper(int *dest) {
  nvshmem_free(dest);
  nvshmem_finalize();
}
