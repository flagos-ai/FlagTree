#include <hip/hip_runtime.h>

// Eager tle.raw sample. One element per thread; the Triton kernel only calls
// this function. extern_func_name is hcu_vector_add.
extern "C" __device__ __attribute__((used)) void hcu_vector_add(float *__restrict__ out, const float *__restrict__ x,
                                          const float *__restrict__ y, int n) {
  int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < n)
    out[i] = x[i] + y[i];
}
