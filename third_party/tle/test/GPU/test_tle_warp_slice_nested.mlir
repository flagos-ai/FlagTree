// RUN: triton-opt %s --verify-diagnostics

#full = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#middle = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#half = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#quarter = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @nested_slice(%src: tensor<256xi32, #full>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %middle = tle.extract_tile %src[%c1] {tile_shape = array<i64: 128>} : tensor<256xi32, #full>, i32 -> tensor<128xi32, #middle>
    %leaf = tle.extract_tile %middle[%c1] {tile_shape = array<i64: 64>} : tensor<128xi32, #middle>, i32 -> tensor<64xi32, #half>
    %tiny = tle.extract_tile %leaf[%c0] {tile_shape = array<i64: 32>} : tensor<64xi32, #half>, i32 -> tensor<32xi32, #quarter>
    tt.return
  }
}
