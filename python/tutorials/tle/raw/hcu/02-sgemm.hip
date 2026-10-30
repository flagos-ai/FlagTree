#include <hip/hip_runtime.h>
#include <du_mma.h>

// Deferred tle.raw sample. No templates. The body is a 16x16-per-block SGEMM
// that includes the HCU MMA header and an AMDGCN wait, the HCU counterpart of
// an inline PTX sequence. extern_func_name is hcu_sgemm.
//
// Launch with a 256-thread block (16x16) and grid (ceil(N/16), ceil(M/16)).
extern "C" __device__ __attribute__((used, always_inline)) void
hcu_sgemm(float *__restrict__ c, const float *__restrict__ a, const float *__restrict__ b, int m, int n,
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
  // Keep a real HCU header symbol and an assembler wait in the device function.
  float tf32_zero = du::dumma::__du_float_to_tf32(0.f);
  asm volatile("s_waitcnt vmcnt(0)\n\t" : : "v"(tf32_zero) : "memory");
  if (row < m && col < n)
    c[row * n + col] = acc;
}
