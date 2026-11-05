extern "C" __device__ __attribute__((noinline)) void
tle_ws_raw_increment(__attribute__((address_space(1))) int *counter) {
  if ((threadIdx.x & 31) == 0)
    atomicAdd(counter, 1);
}
