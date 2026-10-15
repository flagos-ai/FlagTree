#include <stdint.h>

// `out += in` over two shared-memory tiles, returning out's memref descriptor
// so tle_raw can alias the buffer back into the kernel. address_space(3) is
// shared memory on corex just like on NVIDIA, and the flattened
// (allocated, aligned, offset, sizes..., strides...) parameter order is the
// tle_raw signature protocol, so this is the trunk ABI verbatim.
__device__ auto SmemAccumulate(
    __attribute__((address_space(3))) float *out_allocated,
    __attribute__((address_space(3))) float *out_aligned,
    const int64_t out_offset, const int64_t out_size0, const int64_t out_size1,
    const int64_t out_stride0, const int64_t out_stride1,
    __attribute__((address_space(3))) float *in_allocated,
    __attribute__((address_space(3))) float *in_aligned,
    const int64_t in_offset, const int64_t in_size0, const int64_t in_size1,
    const int64_t in_stride0, const int64_t in_stride1) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int64_t numel = out_size0 * out_size1;

  for (int64_t i = tid; i < numel; i += nthreads) {
    const int64_t row = i / out_size1;
    const int64_t col = i % out_size1;
    out_aligned[out_offset + row * out_stride0 + col * out_stride1] +=
        in_aligned[in_offset + row * in_stride0 + col * in_stride1];
  }

  __syncthreads();

  struct {
    __attribute__((address_space(3))) float *allocated;
    __attribute__((address_space(3))) float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  } r{out_allocated,
      out_aligned,
      out_offset,
      {out_size0, out_size1},
      {out_stride0, out_stride1}};
  return r;
}
