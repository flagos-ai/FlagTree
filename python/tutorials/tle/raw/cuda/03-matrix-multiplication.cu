// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdint.h>

__device__ auto
MatMul(__attribute__((address_space(3))) float *output_allocated,
       __attribute__((address_space(3))) float *output_aligned,
       const int64_t output_offsets, const int64_t output_size1,
       const int64_t output_size2, const int64_t output_stride1,
       const int64_t output_stride2,
       __attribute__((address_space(3))) __fp16 *a_allocated,
       __attribute__((address_space(3))) __fp16 *a_aligned,
       const int64_t a_offsets, const int64_t a_size1, const int64_t a_size2,
       const int64_t a_stride1, const int64_t a_stride2,
       __attribute__((address_space(3))) __fp16 *b_allocated,
       __attribute__((address_space(3))) __fp16 *b_aligned,
       const int64_t b_offsets, const int64_t b_size1, const int64_t b_size2,
       const int64_t b_stride1, const int64_t b_stride2) {
  const int idx = threadIdx.x;
  const int bdimx = blockDim.x;
  const int64_t m = output_size1;
  const int64_t n = output_size2;
  const int64_t k = a_size2;

  for (int i = idx; i < m * n; i += bdimx) {
    int row = i / n;
    int col = i % n;
    float acc = 0;
    for (int j = 0; j < k; j++) {
      acc += a_aligned[a_offsets + row * a_stride1 + j * a_stride2] *
             b_aligned[b_offsets + j * b_stride1 + col * b_stride2];
    }
    output_aligned[output_offsets + row * output_stride1 +
                   col * output_stride2] += acc;
  }

  __syncthreads();

  struct {
    __attribute__((address_space(3))) float *allocated;
    __attribute__((address_space(3))) float *aligned;
    int64_t offsets;
    int64_t sizes[2];
    int64_t strides[2];
  } r{output_allocated,
      output_aligned,
      output_offsets,
      {output_size1, output_size2},
      {output_stride1, output_stride2}};
  return r;
}
