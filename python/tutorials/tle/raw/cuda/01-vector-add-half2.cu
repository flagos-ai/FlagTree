#include <cuda_fp16.h>

__device__ void
VectorAddHalf2(__attribute__((address_space(1))) __half *C,
               __attribute__((address_space(1))) const __half *A,
               __attribute__((address_space(1))) const __half *B, const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const __half2 *A2 = reinterpret_cast<const __half2 *>(A);
  const __half2 *B2 = reinterpret_cast<const __half2 *>(B);
  __half2 *C2 = reinterpret_cast<__half2 *>(C);

  for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
    C2[i] = __hadd2(A2[i], B2[i]);
  }

  if (idx == 0 && N % 2 != 0) {
    int last = N - 1;
    C[last] = __hadd(A[last], B[last]);
  }
}
