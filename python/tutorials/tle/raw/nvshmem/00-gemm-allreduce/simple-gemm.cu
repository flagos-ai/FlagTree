

#define TILE_SIZE 16
extern "C" __device__ void tiled_gemm(
    float* C,
    const float* A, 
    const float* B,
    int m, int n, int k
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int thread_row = threadIdx.x / TILE_SIZE;
    int thread_col = threadIdx.x % TILE_SIZE;

    int C_row = block_row * TILE_SIZE + thread_row;
    int C_col = block_col * TILE_SIZE + thread_col;

    float sum = 0.0f;
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int A_col = t * TILE_SIZE + thread_col;
        if (C_row < m && A_col < k) {
            As[thread_row][thread_col] = A[C_row * k + A_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        int B_row = t * TILE_SIZE + thread_row;
        if (B_row < k && C_col < n) {
            Bs[thread_row][thread_col] = B[B_row * n + C_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[thread_row][i] * Bs[i][thread_col];
        }

        __syncthreads();
    }

    if (C_row < m && C_col < n) {
        C[C_row * n + C_col] = sum;
    }
}