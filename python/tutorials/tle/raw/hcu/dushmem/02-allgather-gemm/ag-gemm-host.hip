#include <stdio.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

// Host side of allgather-gemm. DCU exposes the NVSHMEM-style stream
// collectives: dushmemx_float_fcollect_on_stream and
// dushmemx_barrier_all_on_stream. The GEMM itself is the deferred device call.

#define HIP_CHECK(stmt)                                                                            \
  do {                                                                                             \
    hipError_t result = (stmt);                                                                    \
    if (result != hipSuccess) {                                                                    \
      fprintf(stderr, "HIP %s\n", hipGetErrorString(result));                                      \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

extern "C" int ag_gemm_prepare(int m_local, int k, int n, int *mype, int *npes, int *mype_node,
                               hipStream_t *stream, float **a_local, float **a_full, float **b,
                               float **c) {
  if (m_local <= 0 || k <= 0 || n <= 0)
    return -1;
  dushmem_init();
  *mype = dushmem_my_pe();
  *npes = dushmem_n_pes();
  *mype_node = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
  HIP_CHECK(hipSetDevice(*mype_node));
  HIP_CHECK(hipStreamCreate(stream));

  size_t local_elems = (size_t)m_local * k;
  size_t full_elems = local_elems * (size_t)(*npes);
  *a_local = (float *)dushmem_malloc(local_elems * sizeof(float));
  *a_full = (float *)dushmem_malloc(full_elems * sizeof(float));
  HIP_CHECK(hipMalloc(b, (size_t)k * n * sizeof(float)));
  HIP_CHECK(hipMalloc(c, full_elems / k * n * sizeof(float)));
  if (*a_local == nullptr || *a_full == nullptr || *b == nullptr || *c == nullptr)
    return -2;
  return 0;
}

extern "C" int ag_gemm_allgather_on_stream(hipStream_t stream, float *a_full, float *a_local,
                                           const float *shard, int nelems) {
  HIP_CHECK(hipMemcpyAsync(a_local, shard, (size_t)nelems * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
  int rc = dushmemx_float_fcollect_on_stream(DUSHMEM_TEAM_WORLD, a_full, a_local, (size_t)nelems,
                                             stream);
  if (rc != 0)
    return rc;
  dushmemx_barrier_all_on_stream(stream);
  return 0;
}

extern "C" int ag_gemm_copy_b_on_stream(hipStream_t stream, float *dst, const float *src, int nelems) {
  HIP_CHECK(hipMemcpyAsync(dst, src, (size_t)nelems * sizeof(float), hipMemcpyDeviceToDevice, stream));
  return 0;
}

extern "C" int ag_gemm_finish(hipStream_t stream, float *a_local, float *a_full, float *b, float *c) {
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  dushmem_free(a_local);
  dushmem_free(a_full);
  return 0;
}

extern "C" void ag_gemm_finalize(void) { MPI_Finalize(); }
