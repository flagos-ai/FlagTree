#include <cstring>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
    return nvshmemx_cumodule_init(module);
}

extern "C" void ring_bcast_before_launch
(
    int *mype, int *npes, int *mype_in_node, int *n_pes_in_node, 
    cudaStream_t *stream, 
    int **data, int **data_h, uint64_t **psync, 
    int data_len
) {

    nvshmem_init();
    *mype = nvshmem_my_pe();
    *npes = nvshmem_n_pes();
    *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    *n_pes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    cudaSetDevice(*mype_in_node);
    cudaStreamCreate(stream);

    *data = (int *)nvshmem_malloc(sizeof(int) * data_len);
    *data_h = (int *)malloc(sizeof(int) * data_len);
    *psync = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

    for (size_t i = 0; i < data_len; i++) (*data_h)[i] = *mype + i;
    cudaMemcpyAsync(*data, *data_h, sizeof(int) * data_len, cudaMemcpyHostToDevice, *stream);
    nvshmemx_barrier_all_on_stream(*stream);
}

extern "C" void ring_bcast_after_launch
(
    cudaStream_t stream, 
    void *data, void *data_h, void *psync,
    int mype, int npes,
    int data_len
) {
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(data_h, data, sizeof(int) * data_len, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    int *h_ptr = (int *)data_h;
    for (size_t i = 0; i < data_len; i++) {
        if (h_ptr[i] != (int) i)
            printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n", mype, i, h_ptr[i], i, (int) i);
    }

    nvshmem_free(data);
    nvshmem_free(psync);
    free(data_h);

    nvshmem_finalize();
}