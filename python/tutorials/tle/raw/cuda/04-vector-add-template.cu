#include <cuda_fp16.h>

#ifndef VECTOR_ELEM_TYPE
#error "VECTOR_ELEM_TYPE must be provided through @dialect(defines=...)"
#endif

#define CONCAT_IMPL(lhs, rhs) lhs##rhs
#define CONCAT(lhs, rhs) CONCAT_IMPL(lhs, rhs)
#define VECTOR_ADD_FUNC_NAME(type) CONCAT(VectorAdd_, type)

template <typename T>
__device__ __attribute__((always_inline)) void
VectorAddImpl(__attribute__((address_space(1))) T *output,
              __attribute__((address_space(1))) const T *x,
              __attribute__((address_space(1))) const T *y, const int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    output[i] = x[i] + y[i];
}

using VectorElemType = VECTOR_ELEM_TYPE;

extern "C" __device__ void VECTOR_ADD_FUNC_NAME(VECTOR_ELEM_TYPE)(
    __attribute__((address_space(1))) VectorElemType *output,
    __attribute__((address_space(1))) const VectorElemType *x,
    __attribute__((address_space(1))) const VectorElemType *y, const int n) {
  VectorAddImpl<VectorElemType>(output, x, y, n);
}
