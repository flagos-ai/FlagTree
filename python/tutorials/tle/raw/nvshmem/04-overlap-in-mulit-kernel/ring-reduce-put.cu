#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>

extern "C" __device__ void ring_reduce_put_timed(
    int *dst, const int *src, int nreduce,
    uint64_t *signal, uint64_t *put_cycles)
{
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    int block_idx = blockIdx.x;
    int num_blocks = gridDim.x;
    int elems_per_block = nreduce / num_blocks;

    if (elems_per_block * (block_idx + 1) > nreduce) return;

    src += block_idx * elems_per_block;
    dst += block_idx * elems_per_block;
    signal += block_idx;

    if (threadIdx.x == 0) {
        uint64_t t0 = clock64();
        nvshmem_int_put_signal_nbi(
            dst,
            src,
            elems_per_block,
            signal,
            1,
            NVSHMEM_SIGNAL_ADD,
            peer
        );
        uint64_t t1 = clock64();
        put_cycles[block_idx] = t1 - t0;
    }
}

extern "C" __device__ void ring_reduce_wait_timed(
    uint64_t *signal, uint64_t *wait_cycles)
{
    int block_idx = blockIdx.x;
    signal += block_idx;

    if (threadIdx.x == 0) {
        uint64_t t0 = clock64();
        nvshmem_quiet();
        nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, 1);
        uint64_t t1 = clock64();
        wait_cycles[block_idx] = t1 - t0;
    } 
    __syncthreads();
}

extern "C" __device__ void local_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    int block_idx = blockIdx.x;

    A += block_idx * M * N;
    B += block_idx * N * K;
    C += block_idx * M * K;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    for (int row = tid; row < M; row += nthreads) {
        for (int col = 0; col < K; col++) {
            C[row * K + col] = 0.0f;
        }
    }

    for (int row = tid; row < M; row += nthreads) {
        for (int k = 0; k < N; k++) {
            float a_val = A[row * N + k];
            for (int col = 0; col < K; col++) {
                C[row * K + col] += a_val * B[k * K + col];
            }
        }
    }
    __syncthreads();
}
