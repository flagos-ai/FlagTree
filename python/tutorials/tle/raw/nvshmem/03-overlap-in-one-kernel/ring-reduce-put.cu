#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// ============================================================
// 通信函数（分 chunk 逐步发起）：用于方案一 overlap 版本
// 每次只发起第 chunk_idx 个 chunk 的 NBI put+signal，立即返回
// ============================================================
extern "C" __device__ void ring_reduce_put_one_chunk(
    int *dst, const int *src, int nreduce,
    uint64_t *signal, int chunk_size, int chunk_idx)
{
    int mype = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();
    int peer  = (mype + 1) % npes;

    int block_idx       = blockIdx.x;
    int num_blocks      = gridDim.x;
    int elems_per_block = nreduce / num_blocks;

    if (elems_per_block * (block_idx + 1) > nreduce) return;

    src    = src    + block_idx * elems_per_block;
    dst    = dst    + block_idx * elems_per_block;
    signal = signal + block_idx;

    int chunk_elems = chunk_size / sizeof(int);

    if (threadIdx.x == 0) {
        nvshmem_int_put_signal_nbi(
            dst + chunk_idx * chunk_elems,
            src + chunk_idx * chunk_elems,
            chunk_elems,
            signal,
            1,
            NVSHMEM_SIGNAL_ADD,
            peer
        );
    }
}

// ============================================================
// 等待函数：阻塞直到所有 chunk 均已到达（signal >= num_chunks）
// ============================================================
extern "C" __device__ void ring_reduce_put_one_chunk_timed(
    int *dst, const int *src, int nreduce,
    uint64_t *signal, int chunk_size, int chunk_idx,
    uint64_t *put_cycles, int num_chunks)
{
    int mype = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();
    int peer  = (mype + 1) % npes;

    int block_idx       = blockIdx.x;
    int num_blocks      = gridDim.x;
    int elems_per_block = nreduce / num_blocks;

    if (elems_per_block * (block_idx + 1) > nreduce) return;

    src    = src    + block_idx * elems_per_block;
    dst    = dst    + block_idx * elems_per_block;
    signal = signal + block_idx;

    int chunk_elems = chunk_size / sizeof(int);

    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t t0 = clock64();
        nvshmem_int_put_signal_nbi(
            dst + chunk_idx * chunk_elems,
            src + chunk_idx * chunk_elems,
            chunk_elems,
            signal,
            1,
            NVSHMEM_SIGNAL_ADD,
            peer
        );
        uint64_t t1 = clock64();
        put_cycles[block_idx * num_chunks + chunk_idx] = t1 - t0;
    }
    __syncthreads();
}

extern "C" __device__ void ring_reduce_wait(uint64_t *signal, uint64_t expected) {
    int block_idx = blockIdx.x;
    signal = signal + block_idx;

    if (threadIdx.x == 0) {
        nvshmem_quiet();
        nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, expected);
        // *signal = 0;
    }
    __syncthreads();
}

// ============================================================
// 等待函数（等待所有 chunk 到达并 reset signal）：overlap 版本收口
// ============================================================
extern "C" __device__ void ring_reduce_wait_timed(
    uint64_t *signal, uint64_t expected,
    uint64_t *wait_cycles, int num_chunks, int wait_idx)
{
    int block_idx = blockIdx.x;
    signal = signal + block_idx;

    if (threadIdx.x == 0) {
        uint64_t t0 = clock64();
        nvshmem_quiet();
        nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, expected);
        uint64_t t1 = clock64();
        wait_cycles[block_idx * num_chunks + wait_idx] = t1 - t0;
    }
    __syncthreads();
}

// ============================================================
// 本地矩阵乘法计算：C[M×K] = A[M×N] × B[N×K]（模拟重计算）
//
// 参数说明：
//   A, B, C : float* 指针，指向整个 [num_blocks, M, N/K] 数组的首地址
//             本函数自动按 blockIdx.x 偏移，取属于本 block 的 slice
//   M/N/K   : 矩阵维度
//
// 目的：模拟真实 transformer 中的计算密集型操作，
//       使得通信-计算重叠能显著减少端到端延迟
// ============================================================
extern "C" __device__ void local_matmul(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    int block_idx = blockIdx.x;

    // 每个 block 只操作自己的 slice：偏移到 block_idx 对应的子矩阵
    A = A + block_idx * M * N;
    B = B + block_idx * N * K;
    C = C + block_idx * M * K;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    // 线程 tid 负责 C 的第 tid 行（stride 为 nthreads）
    // 内层循环按 col 方向遍历：
    //   - A[row*N + k] 在 k 固定时按 row 变化，线程间 coalesced
    //   - B[k*K + col] 在 col 方向连续，内层按 col 遍历为 coalesced
    //   - C[row*K + col] 内层按 col 连续写，coalesced
    // 先将本线程负责的 C 行清零，确保多次调用时结果正确
    for (int row = tid; row < M; row += nthreads) {
        for (int col = 0; col < K; col++) {
            C[row * K + col] = 0.0f;
        }
    }
    // C[row][col] = sum_k A[row][k] * B[k][col]
    // 外层 k 循环让 a_val 可被寄存器缓存；内层 col 遍历 B/C 的连续内存（coalesced）
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