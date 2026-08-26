#include <cuda_fp16.h>
#include <stdint.h>

using shared_half_ptr = __attribute__((address_space(3))) half *;

__attribute__((device)) auto
AddBeforeDot(shared_half_ptr a_allocated, shared_half_ptr a_aligned,
             const int64_t a_offset, const int64_t a_size0,
             const int64_t a_size1, const int64_t a_stride0,
             const int64_t a_stride1, shared_half_ptr b_allocated,
             shared_half_ptr b_aligned, const int64_t b_offset,
             const int64_t b_size0, const int64_t b_size1,
             const int64_t b_stride0, const int64_t b_stride1) {
  const int64_t tid = threadIdx.x;
  const int64_t num_threads = blockDim.x;
  const half increment = __float2half(0.5f);

  for (int64_t linear = tid; linear < a_size0 * a_size1;
       linear += num_threads) {
    const int64_t row = linear / a_size1;
    const int64_t col = linear % a_size1;
    const int64_t index = a_offset + row * a_stride0 + col * a_stride1;
    a_aligned[index] = __hadd(a_aligned[index], increment);
  }

  for (int64_t linear = tid; linear < b_size0 * b_size1;
       linear += num_threads) {
    const int64_t row = linear / b_size1;
    const int64_t col = linear % b_size1;
    const int64_t index = b_offset + row * b_stride0 + col * b_stride1;
    b_aligned[index] = __hadd(b_aligned[index], increment);
  }

  // Make both updated tiles visible before tl.dot consumes them.
  __syncthreads();

  struct MemRef2D {
    shared_half_ptr allocated;
    shared_half_ptr aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  };
  struct {
    MemRef2D a;
    MemRef2D b;
  } result{
      {a_allocated,
       a_aligned,
       a_offset,
       {a_size0, a_size1},
       {a_stride0, a_stride1}},
      {b_allocated,
       b_aligned,
       b_offset,
       {b_size0, b_size1},
       {b_stride0, b_stride1}},
  };
  return result;
}
