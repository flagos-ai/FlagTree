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


extern "C" void put_block_before_launch
(
    int *mype, int *npes, int *mype_in_node, int *n_pes_in_node, 
    float **send_data, float **recv_data, 
    int num_elems
) {

    nvshmem_init();
    *mype = nvshmem_my_pe();
    *npes = nvshmem_n_pes();
    *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    *n_pes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(*mype_in_node));

    *send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    *recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
}


extern "C" void put_block_after_launch
(
    int mype, int npes,
    void *send_data, void *recv_data,
    int num_elems
) {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /* Do data validation */
    float *host = new float[num_elems];
    CUDA_CHECK(cudaMemcpy(host, recv_data, num_elems * sizeof(float), cudaMemcpyDefault));
    int ref = (mype - 1 + npes) % npes;
    bool success = true;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != ref) {
            printf("Error at %d of rank %d: %f\n", i, mype, host[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[%d of %d] run complete \n", mype, npes);
    } else {
        printf("[%d of %d] run failure \n", mype, npes);
    }

    nvshmem_free(send_data);
    nvshmem_free(recv_data);

    nvshmem_finalize();
}