#include <nvshmem.h>

extern "C" __device__ void simple_shift(int *destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  nvshmem_int_p(destination, mype, peer);
}
