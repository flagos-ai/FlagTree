extern "C" __device__ int nvshmem_my_pe();
extern "C" __device__ int nvshmem_n_pes();
extern "C" __device__ void nvshmem_int_p(int *dest, int value, int pe);

extern "C" __device__ void simple_shift(int *destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  nvshmem_int_p(destination, mype, peer);
}
