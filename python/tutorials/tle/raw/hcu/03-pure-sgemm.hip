#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "02-sgemm.hip"

__global__ void hcu_sgemm_kernel(float *c, const float *a, const float *b, int m, int n, int k) {
  hcu_sgemm(c, a, b, m, n, k);
}

#define CHECK(cmd)                                                                                                     \
  do {                                                                                                                 \
    hipError_t err = (cmd);                                                                                            \
    if (err != hipSuccess) {                                                                                           \
      fprintf(stderr, "HIP error %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(err));                            \
      return 1;                                                                                                        \
    }                                                                                                                  \
  } while (0)

int main(int argc, char **argv) {
  int m = argc > 1 ? atoi(argv[1]) : 1024;
  int n = argc > 2 ? atoi(argv[2]) : 1024;
  int k = argc > 3 ? atoi(argv[3]) : 1024;
  int iters = argc > 4 ? atoi(argv[4]) : 10;
  size_t asz = static_cast<size_t>(m) * k;
  size_t bsz = static_cast<size_t>(k) * n;
  size_t csz = static_cast<size_t>(m) * n;
  float *a;
  float *b;
  float *c;
  CHECK(hipMalloc(&a, asz * sizeof(float)));
  CHECK(hipMalloc(&b, bsz * sizeof(float)));
  CHECK(hipMalloc(&c, csz * sizeof(float)));
  CHECK(hipMemset(a, 0, asz * sizeof(float)));
  CHECK(hipMemset(b, 0, bsz * sizeof(float)));
  dim3 block(256);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  for (int i = 0; i < 3; ++i)
    hcu_sgemm_kernel<<<grid, block>>>(c, a, b, m, n, k);
  CHECK(hipDeviceSynchronize());
  hipEvent_t start, stop;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hcu_sgemm_kernel<<<grid, block>>>(c, a, b, m, n, k);
  CHECK(hipEventRecord(stop));
  CHECK(hipEventSynchronize(stop));
  float ms = 0.f;
  CHECK(hipEventElapsedTime(&ms, start, stop));
  printf("pure_ms %g\n", ms / iters);
  return 0;
}
