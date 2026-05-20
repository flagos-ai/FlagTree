#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"

extern "C" __device__ void set_and_shift(float *send_data, float *recv_data, int num_elems, int mype, int npes) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* set the corresponding element of send_data */
    if (thread_idx < num_elems) send_data[thread_idx] = mype;

    int peer = (mype + 1) % npes;
    /* Every thread in block 0 calls nvshmemx_float_put_block. Alternatively,
       every thread can call shmem_float_p, but shmem_float_p has a disadvantage
       that when the destination GPU is connected via IB, there will be one rma
       message for every single element which can be detrimental to performance.
       And the disadvantage with shmem_float_put is that when the destination GPU is p2p
       connected, it cannot leverage multiple threads to copy the data to the destination
       GPU. */
    int block_offset = blockIdx.x * blockDim.x;
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                             min(blockDim.x, num_elems - block_offset),
                             peer); /* All threads in a block call the API
                                       with the same arguments */
}