#include <math_constants.h>
#include <stdint.h>

__attribute__((device, always_inline)) void
SoftmaxFragment(float *fragment, int64_t fragment_size) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;
  __shared__ float partials[32];

  // Reduce thread-local maxima, then warp maxima, then the CTA maximum.
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int64_t i = 0; i < fragment_size; ++i)
    maximum = fmaxf(maximum, fragment[i]);
  for (int offset = 16; offset > 0; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  if (lane == 0)
    partials[warp] = maximum;
  __syncthreads();
  maximum = lane < num_warps ? partials[lane] : -CUDART_INF_F;
  for (int offset = 16; offset > 0; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  // All warps must finish reading before partials is reused for sums.
  __syncthreads();

  float sum = 0.0f;
#pragma unroll
  for (int64_t i = 0; i < fragment_size; ++i) {
    const float exponent = CUDART_L2E_F * (fragment[i] - maximum);
    float value;
    asm("ex2.approx.f32 %0, %1;" : "=f"(value) : "f"(exponent));
    fragment[i] = value;
    sum += value;
  }
  for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  if (lane == 0)
    partials[warp] = sum;
  __syncthreads();
  sum = lane < num_warps ? partials[lane] : 0.0f;
  for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  __syncthreads();

  const float inverse = 1.0f / sum;
#pragma unroll
  for (int64_t i = 0; i < fragment_size; ++i)
    fragment[i] *= inverse;
}
