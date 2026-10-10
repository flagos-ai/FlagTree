#include <stdio.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

extern "C" int simple_shift_before_launch(int *mype, int *npes, int *mype_node, hipStream_t *stream,
                                           int **destination) {
    dushmem_init();
    *mype = dushmem_my_pe();
    *npes = dushmem_n_pes();
    *mype_node = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
    hipSetDevice(*mype_node);
    hipStreamCreate(stream);
    *destination = (int *)dushmem_malloc(sizeof(int));
    hipMemsetAsync(*destination, 0xFF, sizeof(int), *stream);
    return 0;
}

extern "C" int simple_shift_after_launch(hipStream_t stream, int *destination, int mype, int npes) {
    int msg = -1;
    int expected = (mype - 1 + npes) % npes;
    dushmemx_barrier_all_on_stream(stream);
    hipMemcpyAsync(&msg, destination, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    printf("%d: received message %d\n", mype, msg);
    fflush(stdout);
    dushmem_free(destination);
    return msg == expected ? 0 : 1;
}

extern "C" void simple_shift_finalize(void) {
    // dushmem_finalize unloads the HIP module and SIGSEGVs in this DTK.
    // MPI still has to leave the job cleanly.
    MPI_Finalize();
}
