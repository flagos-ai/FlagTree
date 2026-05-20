#include <cuda.h>

#define THRESHOLD 42
#define CORRECTION 7
extern "C" __device__ void correct_accumulate(int *input, int *partial_sum, int *full_sum) {
    int index = threadIdx.x;
    if (*full_sum > THRESHOLD) {
        input[index] = input[index] - CORRECTION;
    }
    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);
}