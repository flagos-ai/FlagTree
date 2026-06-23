#include <cstring>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
    return nvshmemx_cumodule_init(module);
}

extern "C" void ring_reduce_before_launch
(
    int *mype, int *npes, int *mype_in_node, int *n_pes_in_node, 
    cudaStream_t *stream, 
    int **src, int **dst, int **data_h, uint64_t **signal, 
    int size
) {

    nvshmem_init();
    *mype = nvshmem_my_pe();
    *npes = nvshmem_n_pes();
    *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    *n_pes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(*mype_in_node));
    CUDA_CHECK(cudaStreamCreate(stream));

    *src = (int *)nvshmem_malloc(sizeof(int) * size);
    *dst = (int *)nvshmem_malloc(sizeof(int) * size);
    *data_h = (int *)malloc(sizeof(int) * size);
    *signal = (uint64_t *)nvshmem_calloc(*n_pes_in_node, sizeof(uint64_t));

    for (size_t i = 0; i < size; i++) (*data_h)[i] = i;
    CUDA_CHECK(cudaMemcpyAsync(*src, *data_h, sizeof(int) * size, cudaMemcpyHostToDevice, *stream));
    nvshmemx_barrier_all_on_stream(*stream);
}

extern "C" void ring_reduce_after_launch
(
    cudaStream_t stream, 
    void *src, void *dst, void *data_h, void *signal,
    int mype, int npes,
    int size
) {
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(data_h, dst, sizeof(int) * size, cudaMemcpyDeviceToHost));
    int *data = (int *)data_h;
    for (size_t i = 0; i < size; i++) {
        if (data[i] != (int)i * npes) {
        // if (mype == 0) {
            // printf("PE %d, data[%zu] = %d expected data[%zu] = %d\n", mype, i, data[i], i, (int)i * npes);
            printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n", mype, i, data[i], i, (int)i * npes);
        }
    }

    nvshmem_free(dst);
    nvshmem_free(src);
    nvshmem_free(signal);
    free(data_h);

    nvshmem_finalize();
}