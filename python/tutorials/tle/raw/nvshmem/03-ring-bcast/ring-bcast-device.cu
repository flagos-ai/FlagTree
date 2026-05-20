#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" __device__ void ring_bcast(int *data, int nelem, int root, uint64_t *psync) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    if (mype == root) *psync = 1;

    nvshmem_signal_wait_until(psync, NVSHMEM_CMP_NE, 0);

    if (mype == npes - 1) return;

    nvshmem_int_put(data, data, nelem, peer);
    nvshmem_fence();
    nvshmemx_signal_op(psync, 1, NVSHMEM_SIGNAL_SET, peer);

    *psync = 0;
}