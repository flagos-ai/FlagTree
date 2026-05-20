#include <cuda.h>

extern "C" __device__ void accumulate(int *input, int *partial_sum) {
    int index = threadIdx.x;
    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);
}
