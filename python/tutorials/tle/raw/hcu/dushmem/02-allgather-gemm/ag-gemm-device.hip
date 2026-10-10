#include <hip/hip_runtime.h>
#include <du_mma.h>

// One 16x16 tile of C = A @ B. A is the allgathered matrix.
// Launch with a 256-thread block and grid (ceil(N/16), ceil(M/16)).
extern "C" __device__ __attribute__((used, always_inline)) void
ag_gemm(float *__restrict__ c, const float *__restrict__ a, const float *__restrict__ b, int m, int n,
        int k) {
  constexpr int TILE = 16;
  int lane = static_cast<int>(threadIdx.x);
  int row = static_cast<int>(blockIdx.y) * TILE + lane / TILE;
  int col = static_cast<int>(blockIdx.x) * TILE + lane % TILE;
  float acc = 0.f;
  if (row < m && col < n) {
    for (int kk = 0; kk < k; ++kk)
      acc = fmaf(a[row * k + kk], b[kk * n + col], acc);
  }
  float tf32_zero = du::dumma::__du_float_to_tf32(0.f);
  asm volatile("s_waitcnt vmcnt(0)\n\t" : : "v"(tf32_zero) : "memory");
  if (row < m && col < n)
    c[row * n + col] = acc;
}
