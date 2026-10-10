// Time the same device function the raw kernel calls. One wave, matching num_warps=1.
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>
#include "simple-shift-device.hip"

__global__ void launch_simple_shift(int *destination) { simple_shift(destination); }

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 1000;
    dushmem_init();
    int mype = dushmem_my_pe();
    int mype_node = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
    hipSetDevice(mype_node);
    hipStream_t stream;
    hipStreamCreate(&stream);
    int *destination = (int *)dushmem_malloc(sizeof(int));

    for (int i = 0; i < 10; ++i)
        launch_simple_shift<<<1, 64, 0, stream>>>(destination);
    dushmemx_barrier_all_on_stream(stream);
    hipStreamSynchronize(stream);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, stream);
    for (int i = 0; i < iters; ++i)
        launch_simple_shift<<<1, 64, 0, stream>>>(destination);
    dushmemx_barrier_all_on_stream(stream);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    float ms = 0.f;
    hipEventElapsedTime(&ms, start, stop);
    if (mype == 0)
        printf("pure_ms %f\n", ms / iters);
    fflush(stdout);
    dushmem_free(destination);
    MPI_Finalize();
    return 0;
}
