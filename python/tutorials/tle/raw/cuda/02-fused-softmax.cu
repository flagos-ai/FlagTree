#include <math_constants.h>

__device__ auto
SoftmaxKernel(__attribute__((address_space(3))) float *output_allocated,
              __attribute__((address_space(3))) float *output_aligned,
              const int64_t output_offsets, const int64_t output_size,
              const int64_t output_stride,
              __attribute__((address_space(3))) float *input_allocated,
              __attribute__((address_space(3))) float *input_aligned,
              const int64_t input_offsets, const int64_t input_size,
              const int64_t input_stride) {
  const int idx = threadIdx.x;
  const int bdimx = blockDim.x;
  const int gridx = gridDim.x;
  const int bldx = blockIdx.x;
  for (int i = idx; i < output_size; i += bdimx) {
    output_aligned[i] = input_aligned[i];
  }

  __syncthreads();
  for (int arg = (output_size) >> 1; arg > 0; arg = (arg) >> 1) {
    for (int i = idx; i + arg < output_size; i += bdimx) {
      if (output_aligned[i] < output_aligned[i + arg])
        output_aligned[i] = output_aligned[i + arg];
    }
    __syncthreads();
  }
  float max_val = output_aligned[0];
  for (int i = idx; i < output_size; i += bdimx) {
    float v = CUDART_L2E * (input_aligned[i] - max_val);
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(v));
    output_aligned[i] = y;
  }
  __syncthreads();
  for (int arg = (output_size) >> 1; arg > 0; arg = (arg) >> 1) {
    for (int i = idx; i + arg < output_size; i += bdimx) {
      output_aligned[i] = output_aligned[i] + output_aligned[i + arg];
    }
    __syncthreads();
  }
  float inv = 1.0f / output_aligned[0];
  for (int i = idx; i < output_size; i += bdimx) {
    float v = CUDART_L2E * (input_aligned[i] - max_val);
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(v));
    output_aligned[i] = y * inv;
  }
  __syncthreads();
  struct {
    __attribute__((address_space(3))) float *allocated;
    __attribute__((address_space(3))) float *aligned;
    int64_t offsets;
    int64_t sizes[1];
    int64_t strides[1];
  } r{
      output_allocated, output_aligned,  output_offsets,
      {output_size},    {output_stride},
  };
  return r;
}
