#include <stdint.h>

// Experimental Hopper L2 writeback: one 256-byte bulk TMA store per row.
//
// Unlike tensor-descriptor TMA, cp.async.bulk.global.shared accepts a runtime
// global address. That is required here because every row may target a
// different rank/token/top-k slot. The row pointers are resolved by Triton and
// staged in shared memory before this boundary.
extern "C" __device__ void TleL2TmaScatter(
    const uint64_t values_smem,
    const uint64_t dst_rows_smem,
    const int valid_rows,
    const int cols) {
  const int lane = threadIdx.x & 31;
  const int warp_in_wg = (threadIdx.x >> 5) & 3;
  const int row_bytes = cols * 2;

  asm volatile("bar.sync 9, 128;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

  if (lane == 0) {
    #pragma unroll 1
    for (int j = 0; j < 16; ++j) {
      const int row = warp_in_wg * 16 + j;
      if (row < valid_rows) {
        const uint64_t src_generic =
            values_smem + static_cast<uint64_t>(row) * row_bytes;
        const uint64_t dst_row_generic =
            dst_rows_smem + static_cast<uint64_t>(row) * sizeof(uint64_t);
        uint64_t src_shared;
        uint64_t dst_row_shared;
        uint64_t dst;
        asm volatile(
            "cvta.to.shared.u64 %0, %3;\n"
            "cvta.to.shared.u64 %1, %4;\n"
            "ld.shared.u64 %2, [%1];\n"
            "cp.async.bulk.global.shared::cta.bulk_group "
            "[%2], [%0], %5;"
            : "=l"(src_shared), "=l"(dst_row_shared), "=l"(dst)
            : "l"(src_generic), "l"(dst_row_generic), "r"(row_bytes)
            : "memory");
      }

      // Two groups of at most eight row copies per issuing warp. Waiting here
      // guarantees that the reusable SMEM tile is no longer read by TMA.
      if ((j & 7) == 7) {
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
    }
  }

  asm volatile("bar.sync 9, 128;" ::: "memory");
}
