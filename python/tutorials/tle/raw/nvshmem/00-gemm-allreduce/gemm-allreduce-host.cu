#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

extern "C" int nvshmemx_cumodule_init_wrapper(CUmodule module) {
  return nvshmemx_cumodule_init(module);
}

extern "C" void gemm_allreduce_before_launch
(
    int *mype, int *npes, int *mype_in_node, int *n_pes_in_node, 
    float **d_A, float **d_B, float **d_C,
    float **h_A, float **h_B,
    int M, int N, int K
) {
    nvshmem_init();
    *mype = nvshmem_my_pe();
    *npes = nvshmem_n_pes();
    *mype_in_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    *n_pes_in_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(*mype_in_node));
    *d_A = (float *)nvshmem_malloc(sizeof(float) * M * K);
    *d_B = (float *)nvshmem_malloc(sizeof(float) * K * N);
    *d_C = (float *)nvshmem_malloc(sizeof(float) * M * N);

    *h_A = (float *)malloc(sizeof(float) * M * K);
    *h_B = (float *)malloc(sizeof(float) * K * N);

    float val = static_cast<float>(*mype + 1);
    for (int i = 0; i < M * K; i++) {
        (*h_A)[i] = val;
    }
    for (int i = 0; i < K * N; i++) {
        (*h_B)[i] = val;
    }

    CUDA_CHECK(cudaMemcpy(*d_A, *h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_B, *h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(*d_C, 0, sizeof(float) * M * N));
}

extern "C" void gemm_allreduce_after_launch
(
    int mype, int npes, 
    float *d_A, float *d_B, float *d_C,
    float *h_A, float *h_B,
    int M, int N, int K
) {
    cudaDeviceSynchronize();
    std::vector<float> h_C_local(M * N);
    cudaMemcpy(h_C_local.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float val = static_cast<float>(mype + 1);
    float local_expected = K * val * val;
    std::cout << "PE " << mype << " local GEMM (first 4): ";
    for (int i = 0; i < 4; ++i) printf("%.1f ", h_C_local[i]);
    std::cout << "[expected: " << local_expected << "]\n";

    nvshmem_barrier_all();
    nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, d_C, d_C, M * N);
    nvshmem_barrier_all();
    cudaDeviceSynchronize();
    std::vector<float> h_C_final(M * N);
    cudaMemcpy(h_C_final.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (mype == 0) {
        float expected = 0.0f;
        for (int p = 0; p < npes; ++p) {
            float v = static_cast<float>(p + 1);
            expected += K * v * v;
        }

        std::cout << "\n========================================\n";
        std::cout << "  After AllReduce (PE 0, first 4): ";
        for (int i = 0; i < 4; ++i) printf("%.1f ", h_C_final[i]);
        std::cout << "\n  Expected: " << expected << "\n";
        std::cout << "  (PE0: " << K*1*1 << " + PE1: " << K*2*2 << " = " << expected << ")\n";

        bool correct = true;
        for (int i = 0; i < M * N && correct; ++i) {
            if (std::fabs(h_C_final[i] - expected) > 0.1f) {
                std::cout << "  ERROR at index " << i << ": got " << h_C_final[i]
                          << ", expected " << expected << "\n";
                correct = false;
            }
        }
        if (correct) std::cout << "  ✓ All values correct!\n";
        std::cout << "========================================\n";
    }

    nvshmem_free(d_A);
    nvshmem_free(d_B);
    nvshmem_free(d_C);
    free(h_A);
    free(h_B);

    nvshmem_finalize();
}