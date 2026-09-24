// RUN: triton-opt %s --verify-diagnostics

// A regular full-CTA extract is allowed to change the warp count even when
// the layouts cannot be represented as a register-local owner view. It must
// retain the legacy shared-memory fallback; reduced reuse-default captures are
// checked separately and still require WarpSlice proof.
#full = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#half = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ordinary_cross_warp(%src: tensor<8x16xi32, #full>) {
    %c0 = arith.constant 0 : i32
    %tile = tle.extract_tile %src[%c0] {tile_shape = array<i64: 4, 16>} : tensor<8x16xi32, #full>, i32 -> tensor<4x16xi32, #half>
    tt.return
  }
}
