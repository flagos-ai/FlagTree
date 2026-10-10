#include <stdio.h>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

// Host side of gemm-allreduce. The partial GEMM is the deferred device call.
// dushmemx_float_sum_reduce_on_stream is the DCU stream allreduce.

#define HIP_CHECK(stmt)                                                                            \
  do {                                                                                             \
    hipError_t result = (stmt);                                                                    \
    if (result != hipSuccess) {                                                                    \
      fprintf(stderr, "HIP %s\n", hipGetErrorString(result));                                      \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

extern "C" int gemm_ar_prepare(int m, int k_local, int n, int *mype, int *npes, int *mype_node,
                               hipStream_t *stream, float **a, float **b, float **partial,
                               float **reduced) {
  if (m <= 0 || k_local <= 0 || n <= 0)
    return -1;
  *mype = dushmem_my_pe();
  *npes = dushmem_n_pes();
  *mype_node = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
  HIP_CHECK(hipSetDevice(*mype_node));
  HIP_CHECK(hipStreamCreate(stream));

  size_t c_elems = (size_t)m * n;
  HIP_CHECK(hipMalloc(a, (size_t)m * k_local * sizeof(float)));
  HIP_CHECK(hipMalloc(b, (size_t)k_local * n * sizeof(float)));
  *partial = (float *)dushmem_malloc(c_elems * sizeof(float));
  *reduced = (float *)dushmem_malloc(c_elems * sizeof(float));
  if (*a == nullptr || *b == nullptr || *partial == nullptr || *reduced == nullptr)
    return -2;
  return 0;
}

extern "C" int gemm_ar_copy_on_stream(hipStream_t stream, float *a, float *b, const float *a_src,
                                      const float *b_src, int a_elems, int b_elems) {
  HIP_CHECK(hipMemcpyAsync(a, a_src, (size_t)a_elems * sizeof(float), hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(b, b_src, (size_t)b_elems * sizeof(float), hipMemcpyDeviceToDevice, stream));
  return 0;
}

extern "C" int gemm_ar_sum_reduce_on_stream(hipStream_t stream, float *reduced, float *partial, int nelems) {
  int rc = dushmemx_float_sum_reduce_on_stream(DUSHMEM_TEAM_WORLD, reduced, partial, (size_t)nelems, stream);
  if (rc != 0)
    return rc;
  dushmemx_barrier_all_on_stream(stream);
  return 0;
}

extern "C" int gemm_ar_finish(hipStream_t stream, float *a, float *b, float *partial, float *reduced) {
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  dushmem_free(partial);
  dushmem_free(reduced);
  return 0;
}
