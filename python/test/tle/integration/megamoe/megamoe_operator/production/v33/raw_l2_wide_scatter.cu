#include <stdint.h>

// UserHopper-style L2 writeback for one BM64 x BN128 accumulator tile.
//
// TLE raw cannot currently link a device function whose ABI mixes shared and
// global pointer address spaces. Both arguments are therefore generic shared
// addresses encoded as uint64. Each destination row pointer is staged in
// dst_rows_smem by Triton, so the CUDA boundary does not need any global pointer
// argument at all.
extern "C" __device__ void TleL2WideScatter(
    const uint64_t values_smem,
    const uint64_t dst_rows_smem,
    const int valid_rows,
    const int cols,
    const int bar_id) {
  const int lane = threadIdx.x & 31;
  const int warp_in_wg = (threadIdx.x >> 5) & 3;
  const int row_in_half_warp = lane >> 4;
  const int lane_in_row = lane & 15;
  const int cols_per_lane = cols / 16;

  // Barrier 9 is intentionally outside the 0/2/3 set emitted by the current
  // TLE WS lowering. Only the 128-thread math partition participates.
  asm volatile("bar.sync %0, 128;" :: "r"(bar_id) : "memory");

  #pragma unroll 1
  for (int j = 0; j < 8; ++j) {
    const int row = warp_in_wg * 16 + j * 2 + row_in_half_warp;
    if (row < valid_rows) {
      const int col = lane_in_row * cols_per_lane;
      const int value_offset = row * cols + col;
      const uint64_t value_generic =
          values_smem + static_cast<uint64_t>(value_offset) * 2;
      const uint64_t row_ptr_generic =
          dst_rows_smem + static_cast<uint64_t>(row) * 8;

      uint64_t value_shared;
      uint64_t row_ptr_shared;
      uint64_t dst;
      uint32_t x0, x1, x2, x3;
      asm volatile(
          "cvta.to.shared.u64 %0, %7;\n"
          "cvta.to.shared.u64 %1, %8;\n"
          "ld.shared.u64 %2, [%1];\n"
          "add.u64 %2, %2, %9;\n"
          "ld.shared.v4.b32 {%3, %4, %5, %6}, [%0];\n"
          "st.global.v4.b32 [%2], {%3, %4, %5, %6};"
          : "=l"(value_shared), "=l"(row_ptr_shared), "=l"(dst),
            "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
          : "l"(value_generic), "l"(row_ptr_generic),
            "l"(static_cast<uint64_t>(col) * 2)
          : "memory");
    }
  }

  // Prevent an early half-warp from overwriting the reusable SMEM tile while
  // another half-warp is still issuing its final remote store.
  asm volatile("bar.sync %0, 128;" :: "r"(bar_id) : "memory");
}
