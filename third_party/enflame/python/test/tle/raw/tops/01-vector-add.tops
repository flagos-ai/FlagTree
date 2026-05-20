__device__ void VectorAdd(__attribute__((address_space(1))) float *C,
                          __attribute__((address_space(1))) const float *A,
                          __attribute__((address_space(1))) const float *B,
                          const int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}
