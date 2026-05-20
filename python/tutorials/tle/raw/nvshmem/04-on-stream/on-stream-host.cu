#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"

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


extern "C" void on_stream_before_launch_accumulate
(
    int *mype, int *npes, int *mype_in_node, int *n_pes_in_node, 
    cudaStream_t *stream, 
    int **input, int **partial_sum, int **full_sum, 
    int input_nelems
) {

    nvshmem_init();
    *mype = nvshmem_my_pe();
    *npes = nvshmem_n_pes();
    *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    *n_pes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(*mype_in_node));
    CUDA_CHECK(cudaStreamCreate(stream));

    *input = (int *)nvshmem_malloc(sizeof(int) * input_nelems);
    *partial_sum = (int *)nvshmem_malloc(sizeof(int));
    *full_sum = (int *)nvshmem_malloc(sizeof(int));
}

extern "C" void on_stream_before_launch_correct_accumulate
(
    cudaStream_t *stream,
    int **partial_sum, int **full_sum,
    int to_all_nelems
) {
    nvshmemx_int_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, *full_sum, *partial_sum, to_all_nelems, *stream);
}


extern "C" void on_stream_after_launch_correct_accumulate
(
    int mype, int npes,
    cudaStream_t stream, 
    void *input, void *partial_sum, void *full_sum
) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("[%d of %d] run complete \n", mype, npes);
    CUDA_CHECK(cudaStreamDestroy(stream));

    nvshmem_free(input);
    nvshmem_free(partial_sum);
    nvshmem_free(full_sum);

    nvshmem_finalize();
}