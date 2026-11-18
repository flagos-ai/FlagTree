#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

// Device half of the manual simple_shift. The host launcher stays outside Triton.
extern "C" __device__ __attribute__((used, always_inline)) void simple_shift(int *destination) {
    int mype = dushmem_my_pe();
    int npes = dushmem_n_pes();
    int peer = (mype + 1) % npes;
    dushmem_int_p(destination, mype, peer);
}
