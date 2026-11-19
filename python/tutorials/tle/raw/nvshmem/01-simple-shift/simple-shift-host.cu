#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>

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

extern "C" void simple_shift_before_launch(int *mype, int *npes,
                                           int *mype_in_node, int *npes_in_node,
                                           cudaStream_t *stream, int **dst,
                                           int **data_h) {
  *mype = nvshmem_my_pe();
  *npes = nvshmem_n_pes();
  *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  *npes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

  CUDA_CHECK(cudaSetDevice(*mype_in_node));
  CUDA_CHECK(cudaStreamCreate(stream));

  *dst = (int *)nvshmem_malloc(sizeof(int));
  *data_h = (int *)malloc(sizeof(int));
}

extern "C" void simple_shift_after_launch(cudaStream_t stream, void *dst,
                                          void *data_h, int mype, int npes) {
  nvshmemx_barrier_all_on_stream(stream);
  cudaMemcpyAsync(data_h, dst, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int *data = (int *)data_h;
  printf("%d: received message %d\n", mype, data[0]);

  nvshmem_free(dst);
  free(data_h);
}
