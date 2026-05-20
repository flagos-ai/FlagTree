#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// perform Allreduce using ring
extern "C" __device__ void ring_reduce(int *dst, const int *src, int nreduce, uint64_t *signal,
                            int chunk_size) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    // printf("mype: %d, npes: %d, peer: %d\n", mype, npes, peer);

    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;
    int block_idx = blockIdx.x;
    int elems_per_block = nreduce / num_blocks;

    // Change src, dst, nreduce, signal to what this block is going to process
    // Each CTA will work independently
    if (elems_per_block * (blockIdx.x + 1) > nreduce) return;
    src = src + block_idx * elems_per_block;
    dst = dst + block_idx * elems_per_block;
    nreduce = elems_per_block;
    signal = signal + block_idx;

    int chunk_elems = chunk_size / sizeof(int);
    int num_chunks = nreduce / chunk_elems;

    // reduce phase
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        if (mype != 0) {
            if (thread_id == 0) nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, chunk + 1);

            __syncthreads();
            for (int i = thread_id; i < chunk_elems; i += num_threads) {
                dst[i] = dst[i] + src[i];
            }
            __syncthreads();
        }
        if (thread_id == 0)
            nvshmem_int_put_signal_nbi(dst, (mype == 0) ? src : dst, chunk_elems, signal, 1,
                                       NVSHMEM_SIGNAL_ADD, peer);
        src = src + chunk_elems;
        dst = dst + chunk_elems;
    }

    // Broadcast phase
    dst = dst - num_chunks * chunk_elems;
    if (thread_id == 0) {
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            if (mype < npes - 1) {  // Last pe already has the final result
                nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE,
                                          (mype == 0) ? chunk + 1 : num_chunks + chunk + 1);
            }
            if (mype < npes - 2)
                nvshmem_int_put_signal_nbi(dst, dst, chunk_elems, signal, 1, NVSHMEM_SIGNAL_ADD,
                                           peer);
            dst = dst + chunk_elems;
        }
        *signal = 0;  // reset for next iteration
    }
}