#include <cstring>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>
#include <stddef.h>

// Torch process-group bootstrap. Rank 0 calls dushmem_get_unique_id_bytes,
// the bytes are broadcast, and every rank calls dushmem_init_from_torch_distributed.
// This is the DUSHMEM counterpart of nvshmem/common-host.cu. The CUDA library
// cannot be reused.

extern "C" int dushmem_get_unique_id_bytes(void *uid_buffer, size_t uid_buffer_size) {
  if (uid_buffer == nullptr || uid_buffer_size < sizeof(dushmemx_uniqueid_t))
    return -1;
  dushmemx_uniqueid_t uid = DUSHMEMX_UNIQUEID_INITIALIZER;
  int status = dushmemx_get_uniqueid(&uid);
  if (status != 0)
    return status;
  memcpy(uid_buffer, &uid, sizeof(uid));
  return 0;
}

extern "C" int dushmem_init_from_torch_distributed(int rank, int nranks, int device, void *uid_buffer,
                                                  size_t uid_buffer_size) {
  if (uid_buffer == nullptr || uid_buffer_size < sizeof(dushmemx_uniqueid_t))
    return -1;
  hipError_t hip_status = hipSetDevice(device);
  if (hip_status != hipSuccess)
    return -2;

  dushmemx_uniqueid_t uid = DUSHMEMX_UNIQUEID_INITIALIZER;
  memcpy(&uid, uid_buffer, sizeof(uid));

  dushmemx_init_attr_t attr;
  memset(&attr, 0, sizeof(attr));
  int status = dushmemx_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  if (status != 0)
    return status;
  return dushmemx_init_attr(DUSHMEMX_INIT_WITH_UNIQUEID, &attr);
}
