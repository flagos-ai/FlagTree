// Copyright 2026-     FlagOS Contributors
// Ring shift via native aclshmem_*: each PE reads the next PE's symmetric
// buffer into a local UB tile, then barrier_all.
// library="cannshmem" pulls official cann/shmem headers into the op JIT.

#include "Utils.h"
#include "shmem.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

extern "C" {
// Operands (tle_raw.call order with output_indices=[1]):
//   inputs:  symm (GM), n
//   outputs: out (UB)
__aiv__ void
_mlir_ciface_cannshmem_ring_read(memref_t<__gm__ float, 1> *symm, int64_t n,
                                 memref_t<__ubuf__ float, 1> *out) {
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();
  const int peer = (me + 1) % npes;

  __gm__ float *base = symm->aligned + symm->offset;
  __gm__ float *src = (__gm__ float *)aclshmem_ptr((__gm__ void *)base, peer);
  __ubuf__ float *dst = out->aligned + out->offset;
  // Small N: one aligned GM→UB burst (bytes must be 32B-aligned for MTE).
  const uint32_t nbytes = static_cast<uint32_t>(n * sizeof(float));
  INTRINSIC(copy_gm_to_ubuf_align_b32, dst, src, 0, 1, nbytes, 0, 0, 0, 0);

  aclshmem_barrier_all();
}
}
