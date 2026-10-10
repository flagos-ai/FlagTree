// Manual example: DTK 26.04 DUSHMEM 3.3, each PE writes its id to the next PE.
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

__global__ void simple_shift(int *destination) {
    int mype = dushmem_my_pe();
    int npes = dushmem_n_pes();
    int peer = (mype + 1) % npes;
    dushmem_int_p(destination, mype, peer);
}

int main(void) {
    int mype_node, msg;
    hipStream_t stream;

    dushmem_init();
    mype_node = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
    hipSetDevice(mype_node);
    hipStreamCreate(&stream);

    int *destination = (int *)dushmem_malloc(sizeof(int));

    simple_shift<<<1, 1, 0, stream>>>(destination);
    dushmemx_barrier_all_on_stream(stream);
    hipMemcpyAsync(&msg, destination, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    int expected = (dushmem_my_pe() - 1 + dushmem_n_pes()) % dushmem_n_pes();
    if (msg != expected) {
        fprintf(stderr, "PE %d: received message %d, expected %d\n", dushmem_my_pe(), msg, expected);
        dushmem_global_exit(1);
    }
    printf("%d: received message %d\n", dushmem_my_pe(), msg);

    dushmem_free(destination);
    dushmem_finalize();
    return 0;
}
