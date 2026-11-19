#include <algorithm>
#include <math_constants.h>
#include <stdint.h>

__attribute__((device, always_inline)) auto
SoftmaxKernel(__attribute__((address_space(3))) float *output_allocated,
              __attribute__((address_space(3))) float *output_aligned,
              const int64_t output_offset, const int64_t output_size,
              const int64_t output_stride,
              __attribute__((address_space(3))) float *input_allocated,
              __attribute__((address_space(3))) float *input_aligned,
              const int64_t input_offset, const int64_t input_size,
              const int64_t input_stride, const int num_threads) {
  const int idx = threadIdx.x;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  // The caller passes its compile-time launch size. After inlining, the
  // descriptor size and this value specialize all loop bounds and indices.
  const int num_warps = num_threads / 32;
  auto input = input_aligned + input_offset;
  auto output = output_aligned + output_offset;
  unsigned perthread = output_size / num_threads;
  bool flag = !perthread && idx < output_size;

  float maximum = -CUDART_INF_F;
  int64_t column = idx;
  if (flag)
    maximum = fmaxf(maximum, input[column * input_stride]);
#pragma unroll
  for (int64_t i = 0; i < perthread; ++i) {
    maximum = fmaxf(maximum, input[column * input_stride]);
    column += num_threads;
  }
  for (int offset = 16; offset > 0; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  if (warp < output_size && lane == 0)
    output[warp * output_stride] = maximum;
  __syncthreads();
#pragma unroll
  for (unsigned i = 0; i < std::min(num_warps, int(output_size)); ++i) {
    maximum = fmaxf(maximum, output[i * output_stride]);
  }

  float sum = 0.0f;
  column = idx;
  if (flag) {
    const float exponent =
        CUDART_L2E_F * (input[column * input_stride] - maximum);
    float value;
    asm("ex2.approx.f32 %0, %1;" : "=f"(value) : "f"(exponent));
    input[column * input_stride] = value;
    sum += value;
  }
#pragma unroll
  for (int64_t i = 0; i < perthread; ++i) {
    const float exponent =
        CUDART_L2E_F * (input[column * input_stride] - maximum);
    float value;
    asm("ex2.approx.f32 %0, %1;" : "=f"(value) : "f"(exponent));
    input[column * input_stride] = value;
    sum += value;
    column += num_threads;
  }
  __syncthreads();

  for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  if (warp < output_size && lane == 0)
    output[warp * output_stride] = sum;
  __syncthreads();
  sum = 0.0f;
#pragma unroll
  for (unsigned i = 0; i < std::min(num_warps, int(output_size)); ++i) {
    sum += output[i * output_stride];
  }
  __syncthreads();

  const float scale = 1.0f / sum;
  column = idx;
  if (flag)
    output[column * output_stride] = input[column * input_stride] * scale;
#pragma unroll
  for (int64_t i = 0; i < perthread; ++i) {
    output[column * output_stride] = input[column * input_stride] * scale;
    column += num_threads;
  }
  __syncthreads();

  struct {
    __attribute__((address_space(3))) float *allocated;
    __attribute__((address_space(3))) float *aligned;
    int64_t offsets;
    int64_t sizes[1];
    int64_t strides[1];
  } r{
      output_allocated, output_aligned,  output_offset,
      {output_size},    {output_stride},
  };
  return r;
}
