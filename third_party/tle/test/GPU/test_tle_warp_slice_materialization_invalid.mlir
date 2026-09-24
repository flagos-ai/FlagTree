// RUN: triton-opt %s -split-input-file --verify-diagnostics

// The first extract is valid only in physical warps [2, 4). Expanding it
// back to the full CTA through the ordinary shared relay would also write
// unrelated registers from warps [0, 2) to the same shared addresses.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_reduced_view_expansion(%src: tensor<128xi32, #full>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bottom = tle.extract_tile %src[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    // expected-error @+1 {{cannot materialize a reduced warp slice outside its owning partition}}
    %full = tle.extract_tile %bottom[%c0] {tile_shape = array<i64: 64>} : tensor<64xi32, #half>, i32 -> tensor<64xi32, #full>
    tt.return
  }
}

// -----

// A dynamic nested index cannot prove a register-only view either. Keeping
// the result's reduced warp count does not make the shared relay safe.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_dynamic_reduced_view(%src: tensor<128xi32, #full>, %index: i32) {
    %c1 = arith.constant 1 : i32
    %bottom = tle.extract_tile %src[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    // expected-error @+1 {{cannot materialize a reduced warp slice outside its owning partition}}
    %tile = tle.extract_tile %bottom[%index] {tile_shape = array<i64: 32>} : tensor<64xi32, #half>, i32 -> tensor<32xi32, #half>
    tt.return
  }
}

// -----

// A static extraction that changes lane ownership also requires a relay.
#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#changed = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_reduced_view_lane_change(%src: tensor<128xi32, #full>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bottom = tle.extract_tile %src[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #full>, i32 -> tensor<64xi32, #half>
    // expected-error @+1 {{cannot materialize a reduced warp slice outside its owning partition}}
    %tile = tle.extract_tile %bottom[%c0] {tile_shape = array<i64: 64>} : tensor<64xi32, #half>, i32 -> tensor<64xi32, #changed>
    tt.return
  }
}
