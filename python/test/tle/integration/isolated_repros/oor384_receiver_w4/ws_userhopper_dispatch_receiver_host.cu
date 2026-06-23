#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" void userhopper_ws_nvshmem_init_wrapper() { nvshmem_init(); }

extern "C" int userhopper_ws_nvshmemx_cumodule_init_wrapper(CUmodule module) {
  return nvshmemx_cumodule_init(module);
}

extern "C" int userhopper_ws_nvshmem_team_mype_wrapper() {
  return nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
}

extern "C" int userhopper_ws_nvshmem_n_pes_wrapper() { return nvshmem_n_pes(); }

extern "C" unsigned char *userhopper_ws_nvshmem_alloc_bytes_wrapper(long long bytes) {
  return reinterpret_cast<unsigned char *>(nvshmem_malloc(static_cast<size_t>(bytes)));
}

extern "C" void userhopper_ws_nvshmem_barrier_all_wrapper() { nvshmem_barrier_all(); }

extern "C" void userhopper_ws_nvshmem_finalize_wrapper(unsigned char *ptr) {
  nvshmem_free(ptr);
  nvshmem_finalize();
}
