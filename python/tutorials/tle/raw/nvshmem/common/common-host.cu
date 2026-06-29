#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
  return nvshmemx_cumodule_init(module);
}

extern "C" void nvshmem_barrier_all_wrapper() { nvshmem_barrier_all(); }

extern "C" int nvshmem_get_unique_id_bytes(void *uid_buffer,
                                           size_t uid_buffer_size) {
  if (uid_buffer_size < sizeof(nvshmemx_uniqueid_t)) {
    return -1;
  }

  nvshmemx_uniqueid_t uid;
  nvshmemx_get_uniqueid(&uid);
  memcpy(uid_buffer, &uid, sizeof(uid));
  return 0;
}

extern "C" int nvshmem_init_from_torch_distributed(int rank, int nranks,
                                                   int cuda_device,
                                                   void *uid_buffer,
                                                   size_t uid_buffer_size) {
  if (uid_buffer_size < sizeof(nvshmemx_uniqueid_t)) {
    return -1;
  }

  CUDA_CHECK(cudaSetDevice(cuda_device));

  nvshmemx_uniqueid_t uid;
  memcpy(&uid, uid_buffer, sizeof(uid));

  nvshmemx_init_attr_t attr;
  memset(&attr, 0, sizeof(attr));
  nvshmemx_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  return 0;
}

extern "C" int nvshmem_finalize_from_torch_distributed() {
  nvshmem_finalize();
  return 0;
}
