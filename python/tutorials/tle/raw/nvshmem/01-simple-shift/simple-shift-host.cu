#include <cstring>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>

extern "C" void nvshmem_init_wrapper() { nvshmem_init(); }

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
  return nvshmemx_cumodule_init(module);
}

extern "C" int nvshmem_team_mype_wrapper() {
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  return mype_node;
}

// TODO: Adapt to different data types
extern "C" int *nvshmem_alloc_wrapper(int size) {
  int *destination = (int *)nvshmem_malloc(sizeof(int) * size);
  return destination;
}

extern "C" void nvshmemx_barrier_warpper(cudaStream_t stream) {
  nvshmemx_barrier_all_on_stream(stream);
}

extern "C" void nvshmem_finalize_wrapper(int *dest) {
  nvshmem_free(dest);
  nvshmem_finalize();
}
